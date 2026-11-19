"""Tests for PotentialManager.

Tests cover the 7 main use cases via the core methods:
1. Single-component: Tip standalone (for amorphisation)
2. Single-component: Substrate standalone (for amorphisation)
3. Single-component: Sheet standalone (single/multi-layer)
4. Langevin-ready: Tip with 3 region types
5. Langevin-ready: Substrate with 3 region types
6. Full system: AFM (tip + sheet + substrate)
7. Full system: Sheet-on-sheet (3-4 layers with ghost interactions)

The builders (AFMBuilder, SheetOnSheetBuilder) call these core methods directly.
Tests verify the core methods work correctly for all use cases.

"""

import pytest
import tempfile
from pathlib import Path

from src.core.potential_manager import (
    PotentialManager,
    POTENTIALS_WITH_INTERNAL_LJ,
    POTENTIALS_REQUIRING_LJ,
)
from src.core.config import ComponentConfig, SheetConfig, TipConfig


# =========================================================================
# Golden-block helpers (MLIP emission tests)
# =========================================================================

def _fake_model(tmp_path, name):
    """Write a non-UTF-8 stub standing in for a serialized MLIP model."""
    p = tmp_path / name
    p.write_bytes(b"\x00")
    return str(p)


def _fake_cif(tmp_path, name):
    """Write a stub CIF so ComponentConfig path resolution keeps the path."""
    p = tmp_path / name
    p.write_text('# dummy', encoding='utf-8')
    return str(p)


def make_component_config(tmp_path, pot_type, pot_path, kind='sheet'):
    """Build a minimal sheet/tip config for PotentialManager tests."""
    if kind == 'tip':
        return TipConfig(mat='Si', pot_type=pot_type, pot_path=pot_path,
                         cif_path=_fake_cif(tmp_path, 'Si.cif'),
                         r=20.0, amorph='c', dspring=0.1)
    # lat_c is mandatory for MLIP sheets (enforced by SheetConfig), so it is
    # supplied here to keep this helper usable for classical and MLIP alike.
    return SheetConfig(mat='h-MoS2', pot_type=pot_type, pot_path=pot_path,
                       cif_path=_fake_cif(tmp_path, 'MoS2.cif'),
                       x=50.0, y=50.0, layers=[1], lat_c=6.15)


def make_sub_config(tmp_path, pot_type, pot_path):
    """Minimal substrate config (Si), for multi-component cell tests."""
    from src.core.config import SubstrateConfig
    return SubstrateConfig(mat='Si', pot_type=pot_type, pot_path=pot_path,
                           cif_path=_fake_cif(tmp_path, 'Si.cif'),
                           thickness=10.0)


def write_and_read(pm, tmp_path, name='system.in.settings'):
    """Render the settings file and return its text."""
    out = Path(tmp_path) / name
    pm.write_file(out)
    return out.read_text(encoding='utf-8')


@pytest.fixture
def mock_structure_readers(monkeypatch):
    """Stub cifread/count_atomtypes: Si.cif -> Si, anything else -> MoS2."""
    def mock_cif(path):
        if Path(str(path)).name.startswith('Si'):
            return {'elements': ['Si']}
        return {'elements': ['Mo', 'S']}

    def mock_count(_path, elements, pot_type=None):
        return {el: 1 for el in elements}

    monkeypatch.setattr('src.core.potential_manager.cifread', mock_cif)
    monkeypatch.setattr('src.core.potential_manager.count_atomtypes', mock_count)


class TestPotentialManagerInit:
    """Tests for PotentialManager initialization."""

    def test_init_with_langevin_settings(self, mock_settings):
        """PM should detect Langevin from settings."""
        mock_settings.thermostat.type = 'langevin'
        pm = PotentialManager(mock_settings)
        assert pm.use_langevin is True

    def test_init_with_nose_hoover_settings(self, mock_settings):
        """PM should not use Langevin with Nose-Hoover thermostat."""
        mock_settings.thermostat.type = 'nose-hoover'
        pm = PotentialManager(mock_settings)
        assert pm.use_langevin is False

    def test_init_with_explicit_langevin_override(self, mock_settings):
        """PM should respect explicit use_langevin override."""
        mock_settings.thermostat.type = 'langevin'
        pm = PotentialManager(mock_settings, use_langevin=False)
        assert pm.use_langevin is False

    def test_lj_cutoff_property(self, mock_settings):
        """PM should expose lj_cutoff from settings via property."""
        pm = PotentialManager(mock_settings)
        assert pm.lj_cutoff == mock_settings.potential.lj_cutoff
        assert pm.lj_cutoff == 11.0


class TestIsSheetLJ:
    """Tests for is_sheet_lj potential classification."""

    def test_potentials_with_internal_lj(self, mock_settings):
        """Potentials with internal LJ should return False."""
        pm = PotentialManager(mock_settings)
        for pot in POTENTIALS_WITH_INTERNAL_LJ:
            assert pm.is_sheet_lj(pot) is False
            assert pm.is_sheet_lj(pot.upper()) is False  # Case insensitive

    def test_potentials_requiring_lj(self, mock_settings):
        """Potentials requiring explicit LJ should return True."""
        pm = PotentialManager(mock_settings)
        for pot in POTENTIALS_REQUIRING_LJ:
            assert pm.is_sheet_lj(pot) is True
            assert pm.is_sheet_lj(pot.upper()) is True

    def test_unknown_potential_defaults_to_true(self, mock_settings):
        """Unknown potentials should default to requiring LJ (safe default)."""
        pm = PotentialManager(mock_settings)
        assert pm.is_sheet_lj('unknown_potential') is True

    def test_is_sheet_lj_is_false_for_every_mlip(self, mock_settings):
        """An MLIP never takes the per-layer LJ path (v4 spec §3).

        The per-layer topology needed NULL-padded type maps, which both
        frameworks reject, so the single full-coverage instance is the only
        answer there is -- and it supplies the interlayer physics itself.
        """
        pm = PotentialManager(mock_settings, use_langevin=False)
        assert pm.is_sheet_lj("deepmd") is False
        assert pm.is_sheet_lj("mace") is False
        assert pm.is_sheet_lj("DeePMD") is False     # case-insensitive
        assert pm.is_sheet_lj("sw") is True          # unchanged
        assert pm.is_sheet_lj("rebomos") is False    # unchanged

    def test_is_sheet_lj_takes_no_interlayer_mode(self, mock_settings):
        """The parameter is gone; a caller still passing it must fail loudly."""
        import inspect

        pm = PotentialManager(mock_settings, use_langevin=False)
        assert 'interlayer_mode' not in inspect.signature(pm.is_sheet_lj).parameters
        assert ('interlayer_mode'
                not in inspect.signature(pm.register_component).parameters)


class TestNormalizationGuards:
    """Tests for defensive normalization inside PotentialManager."""

    def test_register_component_normalizes_potential_usage_key(
        self,
        mock_settings,
        monkeypatch,
    ):
        """Mixed-case pot_type should map to a single lowercase usage key."""
        # Monkeypatch path validation functions in the modules that use them
        monkeypatch.setattr(
            'src.core.config.get_potential_path',
            lambda _path: Path(_path)
        )
        monkeypatch.setattr(
            'src.core.config.get_material_path',
            lambda _path: Path(_path)
        )
        monkeypatch.setattr(
            'src.core.potential_manager.cifread',
            lambda _path: {'elements': ['Si']}
        )
        monkeypatch.setattr(
            'src.core.potential_manager.count_atomtypes',
            lambda _path, _elements, pot_type=None: {'Si': 1}
        )

        from src.core.config import TipConfig
        config_upper = TipConfig(
            mat='Si', pot_type='SW', pot_path='/tmp/Si.sw', cif_path='/tmp/Si.cif',
            r=20.0, amorph='c', dspring=0.1, s=1.0
        )
        pm = PotentialManager(mock_settings, use_langevin=False)
        pm.register_component('tip', config_upper)

        assert 'sw' in pm.potential_usage
        assert 'SW' not in pm.potential_usage

    def test_add_self_interaction_uses_normalized_key_for_indexing(
        self,
        mock_settings,
        monkeypatch,
    ):
        """Potential indexing should stay consistent when pot_type casing differs."""
        # Monkeypatch path validation functions in the modules that use them
        monkeypatch.setattr(
            'src.core.config.get_potential_path',
            lambda _path: Path(_path)
        )
        monkeypatch.setattr(
            'src.core.config.get_material_path',
            lambda _path: Path(_path)
        )
        monkeypatch.setattr(
            'src.core.potential_manager.cifread',
            lambda _path: {'elements': ['Si']}
        )
        monkeypatch.setattr(
            'src.core.potential_manager.count_atomtypes',
            lambda _path, _elements, pot_type=None: {'Si': 1}
        )

        # Create minimal configs without needing real files
        from src.core.config import TipConfig, SubstrateConfig
        config_upper = TipConfig(
            mat='Si', pot_type='SW', pot_path='/tmp/Si.sw', cif_path='/tmp/Si.cif',
            r=20.0, amorph='c', dspring=0.1, s=1.0
        )
        config_lower = SubstrateConfig(
            mat='Si', pot_type='sw', pot_path='/tmp/Si.sw', cif_path='/tmp/Si.cif',
            thickness=10.0, amorph='c'
        )

        pm = PotentialManager(mock_settings, use_langevin=False)
        pm.register_component('tip', config_upper)
        pm.register_component('sub', config_lower)

        pm.add_self_interaction('tip')
        pm.add_self_interaction('sub')

        assert len(pm.self_interaction_commands) == 2
        assert ' sw ' in pm.self_interaction_commands[0].lower()
        assert ' sw ' in pm.self_interaction_commands[1].lower()
        assert 'sw 1' in pm.self_interaction_commands[0].lower()
        assert 'sw 2' in pm.self_interaction_commands[1].lower()

    def test_mixed_case_elements_use_override_in_cross_interaction(self, mock_settings):
        """Mixed-case element symbols should still resolve override parameters."""
        pm = PotentialManager(mock_settings, use_langevin=False)
        pm.set_lj_overrides({'mo-s': [0.5, 3.5]})

        pm.types.register(component='a', element='MO', pot_label='MO')
        pm.types.register(component='b', element='s', pot_label='s')
        pm.add_cross_interaction('a', 'b')

        command = pm.cross_interaction_commands[0]
        assert 'lj/cut 0.5 3.5' in command


class TestLJOverrides:
    """Tests for user-provided LJ override behavior."""

    def test_set_lj_overrides_parses_pair_keys(self, mock_settings):
        """Pair keys should accept multiple separators and normalize order."""
        pm = PotentialManager(mock_settings, use_langevin=False)
        pm.set_lj_overrides({
            'Mo-S': [0.4124, 3.75114],
            'S_Mo': [0.4999, 3.9],
        })

        # Last assignment for the same canonical pair wins
        assert pm.lj_overrides[('Mo', 'S')] == (0.4999, 3.9)

    def test_set_lj_overrides_ignores_file_key(self, mock_settings):
        """A literal 'file' key reaching here (a config built without going
        through expand_config_sweeps/resolve_lj_override_file) must be
        skipped, not handed to _parse_override_pair and crash."""
        pm = PotentialManager(mock_settings, use_langevin=False)
        pm.set_lj_overrides({'file': 'lj.txt', 'Mo-S': [0.4, 3.7]})

        assert len(pm.lj_overrides) == 1
        assert pm.lj_overrides[('Mo', 'S')] == (0.4, 3.7)

    def test_interlayer_uses_override_values(
        self, mock_settings, mos2_sheet_config
    ):
        """Interlayer pair_coeff should use override epsilon/sigma for matching pairs."""
        import src.core.potential_manager as pm_module

        def mock_cifread(path):
            return {'elements': ['Mo', 'S']}

        def mock_count(path, elements, pot_type=None):
            return {el: 1 for el in elements}

        original_cifread = pm_module.cifread
        original_count = pm_module.count_atomtypes
        pm_module.cifread = mock_cifread
        pm_module.count_atomtypes = mock_count

        try:
            pm = PotentialManager(mock_settings, use_langevin=False)
            pm.set_lj_overrides({
                'Mo-Mo': [1.0624, 3.878597],
                'Mo-S': [0.4124, 3.75114],
                'S-S': [0.198443, 3.62368],
            })
            pm.register_component('sheet', mos2_sheet_config, n_layers=2)
            pm.add_interlayer_interaction('sheet')

            joined = "\n".join(pm.cross_interaction_commands)
            assert 'lj/cut 1.0624 3.878597' in joined
            assert 'lj/cut 0.4124 3.75114' in joined
            assert 'lj/cut 0.198443 3.62368' in joined
        finally:
            pm_module.cifread = original_cifread
            pm_module.count_atomtypes = original_count


# =========================================================================
# Use Case 1: Single-component Tip (amorphisation)
# =========================================================================

class TestSingleComponentTip:
    """Tests for standalone tip simulation setup."""

    def test_tip_standard_types(
        self, mock_settings, si_tip_config, mock_cifread_si, mock_count_atomtypes_si
    ):
        """Single Si tip should get 1 atom type."""
        mock_settings.thermostat.type = 'nose-hoover'  # No Langevin
        pm = PotentialManager(mock_settings, use_langevin=False)
        
        # Use core method
        result = pm.register_component('tip', si_tip_config)
        pm.add_self_interaction('tip')
        
        assert 'Si' in result
        assert len(result['Si']) == 1
        assert result['Si'] == [1]
        assert len(pm.types) == 1

    def test_tip_self_interaction_only(
        self, mock_settings, si_tip_config, mock_cifread_si, mock_count_atomtypes_si
    ):
        """Single tip should have self-interaction but no cross-interactions."""
        pm = PotentialManager(mock_settings, use_langevin=False)
        pm.register_component('tip', si_tip_config)
        pm.add_self_interaction('tip')
        
        assert len(pm.self_interaction_commands) == 1
        assert len(pm.cross_interaction_commands) == 0
        assert 'tip' in pm.self_interaction_commands[0]

    def test_tip_group_string(
        self, mock_settings, si_tip_config, mock_cifread_si, mock_count_atomtypes_si
    ):
        """Group string should contain all tip types."""
        pm = PotentialManager(mock_settings, use_langevin=False)
        pm.register_component('tip', si_tip_config)
        
        group_str = pm.types.get_group_string('tip')
        assert group_str == '1'


# =========================================================================
# Use Case 2: Single-component Substrate (amorphisation)
# =========================================================================

class TestSingleComponentSubstrate:
    """Tests for standalone substrate simulation setup."""

    def test_sub_standard_types(
        self, mock_settings, si_sub_config, mock_cifread_si, mock_count_atomtypes_si
    ):
        """Single Si substrate should get 1 atom type."""
        pm = PotentialManager(mock_settings, use_langevin=False)
        
        result = pm.register_component('sub', si_sub_config)
        pm.add_self_interaction('sub')
        
        assert 'Si' in result
        assert len(result['Si']) == 1
        assert len(pm.types) == 1

    def test_sub_no_cross_interactions(
        self, mock_settings, si_sub_config, mock_cifread_si, mock_count_atomtypes_si
    ):
        """Standalone substrate should have no cross-interactions."""
        pm = PotentialManager(mock_settings, use_langevin=False)
        pm.register_component('sub', si_sub_config)
        pm.add_self_interaction('sub')
        
        assert len(pm.cross_interaction_commands) == 0


# =========================================================================
# Use Case 3: Single-component Sheet (single and multi-layer)
# =========================================================================

class TestSingleComponentSheet:
    """Tests for standalone sheet simulation setup."""

    def test_sheet_single_layer(
        self, mock_settings, mos2_sheet_config, mock_cifread_mos2, mock_count_atomtypes_mos2_simple
    ):
        """Single-layer sheet should get types for each element."""
        pm = PotentialManager(mock_settings, use_langevin=False)
        
        result = pm.register_component('sheet', mos2_sheet_config, n_layers=1)
        pm.add_self_interaction('sheet')
        
        assert 'Mo' in result
        assert 'S' in result
        assert len(pm.types) == 2

    def test_sheet_multi_layer_types(
        self, mock_settings, mos2_sheet_config, mock_cifread_mos2, mock_count_atomtypes_mos2_simple
    ):
        """Multi-layer sheet should get types per layer."""
        pm = PotentialManager(mock_settings, use_langevin=False)
        
        result = pm.register_component('sheet', mos2_sheet_config, n_layers=2)
        
        # 2 elements × 2 layers = 4 types
        assert len(pm.types) == 4

    def test_sheet_multi_layer_interlayer_lj(
        self, mock_settings, mos2_sheet_config, mock_cifread_mos2, mock_count_atomtypes_mos2_simple
    ):
        """Multi-layer sheet should have interlayer LJ interactions."""
        pm = PotentialManager(mock_settings, use_langevin=False)
        pm.register_component('sheet', mos2_sheet_config, n_layers=2)
        pm.add_self_interaction('sheet')
        pm.add_interlayer_interaction('sheet')
        
        # Should have interlayer LJ commands (Mo-Mo, Mo-S, S-Mo, S-S between layers)
        assert len(pm.cross_interaction_commands) > 0


# =========================================================================
# Use Case 4 & 5: Langevin-ready Components (Tip & Substrate)
# =========================================================================

class TestLangevinComponents:
    """Tests for Langevin thermostat region setup."""

    def test_tip_langevin_3x_types(
        self, mock_settings, si_tip_config, mock_cifread_si, mock_count_atomtypes_si
    ):
        """Tip with Langevin should get 3 types per element."""
        pm = PotentialManager(mock_settings, use_langevin=True)  # Enable Langevin
        
        result = pm.register_component('tip', si_tip_config)
        
        # Si has 1 pot type × 3 regions = 3 types
        assert 'Si' in result
        assert len(result['Si']) == 3
        assert len(pm.types) == 3

    def test_sub_langevin_3x_types(
        self, mock_settings, si_sub_config, mock_cifread_si, mock_count_atomtypes_si
    ):
        """Substrate with Langevin should get 3 types per element."""
        pm = PotentialManager(mock_settings, use_langevin=True)
        
        result = pm.register_component('sub', si_sub_config)
        
        assert len(result['Si']) == 3
        assert len(pm.types) == 3

    def test_sheet_no_langevin_expansion(
        self, mock_settings, mos2_sheet_config, mock_cifread_mos2, mock_count_atomtypes_mos2_simple
    ):
        """Sheets should never get Langevin expansion regardless of PM setting."""
        pm = PotentialManager(mock_settings, use_langevin=True)
        
        result = pm.register_component('sheet', mos2_sheet_config)
        
        # Should only have 2 types (Mo, S), not 6 (Mo×3, S×3)
        assert len(pm.types) == 2

    def test_langevin_group_definitions(
        self, mock_settings, si_tip_config, mock_cifread_si, mock_count_atomtypes_si
    ):
        """Langevin types should have proper group definitions (mobile, fix, thermo)."""
        pm = PotentialManager(mock_settings, use_langevin=True)
        pm.register_component('tip', si_tip_config)
        
        # Check generated group names from registered atom types
        group_names = [atype.group_name for atype in pm.types]
        
        # Should have tip_t1, tip_fix_t1, tip_thermo_t1
        assert any('tip_t' in g and '_fix' not in g and '_thermo' not in g for g in group_names)
        assert any('_fix' in g for g in group_names)
        assert any('_thermo' in g for g in group_names)


# =========================================================================
# Use Case 6: AFM System (Tip + Sheet + Substrate)
# =========================================================================

class TestAFMSystem:
    """Tests for complete AFM system setup."""

    def test_afm_all_components_registered(
        self, mock_settings, si_tip_config, si_sub_config, mos2_sheet_config,
        mock_cifread_si, mock_count_atomtypes_si
    ):
        """AFM system should register all three components."""
        # Need to mock both Si and MoS2 cifread
        import src.core.potential_manager as pm_module
        
        def mock_cifread(path):
            if 'MoS2' in str(path) or 'h-Mo' in str(path):
                return {'elements': ['Mo', 'S']}
            return {'elements': ['Si']}
        
        def mock_count(path, elements, pot_type=None):
            return {el: 1 for el in elements}
        
        original_cifread = pm_module.cifread
        original_count = pm_module.count_atomtypes
        pm_module.cifread = mock_cifread
        pm_module.count_atomtypes = mock_count
        
        try:
            pm = PotentialManager(mock_settings, use_langevin=False)
            # Register all components
            pm.register_component('sub', si_sub_config)
            pm.register_component('tip', si_tip_config)
            pm.register_component('sheet', mos2_sheet_config, n_layers=1)
            
            # Add self-interactions
            pm.add_self_interaction('sub')
            pm.add_self_interaction('tip')
            pm.add_self_interaction('sheet')
            
            # Si(sub) + Si(tip) + Mo + S = 4 types
            assert len(pm.types) == 4
        finally:
            pm_module.cifread = original_cifread
            pm_module.count_atomtypes = original_count

    def test_afm_cross_interactions(
        self, mock_settings, si_tip_config, si_sub_config, mos2_sheet_config
    ):
        """AFM system should have cross-interactions between all pairs."""
        import src.core.potential_manager as pm_module
        
        def mock_cifread(path):
            if 'MoS2' in str(path) or 'h-Mo' in str(path):
                return {'elements': ['Mo', 'S']}
            return {'elements': ['Si']}
        
        def mock_count(path, elements, pot_type=None):
            return {el: 1 for el in elements}
        
        original_cifread = pm_module.cifread
        original_count = pm_module.count_atomtypes
        pm_module.cifread = mock_cifread
        pm_module.count_atomtypes = mock_count
        
        try:
            pm = PotentialManager(mock_settings, use_langevin=False)
            # Register
            pm.register_component('sub', si_sub_config)
            pm.register_component('tip', si_tip_config)
            pm.register_component('sheet', mos2_sheet_config, n_layers=1)
            
            # Self-interactions
            pm.add_self_interaction('sub')
            pm.add_self_interaction('tip')
            pm.add_self_interaction('sheet')
            
            # Cross-interactions (like AFMBuilder does)
            pm.add_cross_interaction('sub', 'tip')
            pm.add_cross_interaction('sub', 'sheet')
            pm.add_cross_interaction('tip', 'sheet')
            
            # Should have LJ cross-interaction commands
            assert len(pm.cross_interaction_commands) >= 3
            
            # Check all component pairs are covered
            cross_str = ' '.join(pm.cross_interaction_commands)
            assert 'lj' in cross_str.lower()
        finally:
            pm_module.cifread = original_cifread
            pm_module.count_atomtypes = original_count


# =========================================================================
# Use Case 7: Sheet-on-Sheet System
# =========================================================================

class TestSheetOnSheetSystem:
    """Tests for sheet-on-sheet friction simulation setup."""

    def test_sheet_on_sheet_4_layers(
        self, mock_settings, mos2_multilayer_config, mock_cifread_mos2, mock_count_atomtypes_mos2_simple
    ):
        """Sheet-on-sheet should create 4 layers with separate types."""
        pm = PotentialManager(mock_settings, use_langevin=False)
        
        pm.register_component('sheet', mos2_multilayer_config, n_layers=4)
        pm.add_self_interaction('sheet')
        
        # 2 elements × 4 layers = 8 types
        assert len(pm.types) == 8

    def test_sheet_on_sheet_layer_groups(
        self, mock_settings, mos2_multilayer_config, mock_cifread_mos2, mock_count_atomtypes_mos2_simple
    ):
        """Each layer should have its own group of types."""
        pm = PotentialManager(mock_settings, use_langevin=False)
        pm.register_component('sheet', mos2_multilayer_config, n_layers=4)
        pm.add_self_interaction('sheet')
        
        for layer in range(4):
            layer_types = pm.types.get_layer_group_string('sheet', layer)
            assert layer_types != ''
            # Each layer should have 2 types (Mo, S)
            assert len(layer_types.split()) == 2

    def test_sheet_on_sheet_ghost_interactions(
        self, mock_settings, mos2_multilayer_config, mock_cifread_mos2, mock_count_atomtypes_mos2_simple
    ):
        """Non-adjacent layers should get ghost (low epsilon) interactions."""
        pm = PotentialManager(mock_settings, use_langevin=False)
        pm.register_component('sheet', mos2_multilayer_config, n_layers=4)
        pm.add_self_interaction('sheet')
        pm.add_ghost_lj('sheet', max_real_distance=1)
        
        # Check we have cross-interaction commands
        assert len(pm.cross_interaction_commands) > 0
        
        # Some should be ghost (1e-100 epsilon)
        ghost_commands = [c for c in pm.cross_interaction_commands 
                         if '1e-100' in c.lower() or '1.00e-100' in c.lower()]
        real_commands = [c for c in pm.cross_interaction_commands 
                        if '1e-100' not in c.lower() and '1.00e-100' not in c.lower()]
        
        # With 4 layers and max_real_distance=1:
        # Real: L1-L2, L2-L3, L3-L4 (3 pairs)
        # Ghost: L1-L3, L1-L4, L2-L4 (3 pairs)
        # Each pair has element combinations: Mo-Mo, Mo-S, S-Mo, S-S (4 combos)
        assert len(ghost_commands) > 0
        assert len(real_commands) > 0

    def test_sheet_on_sheet_layer_groups_string(
        self, mock_settings, mos2_multilayer_config, mock_cifread_mos2, mock_count_atomtypes_mos2_simple
    ):
        """Layer groups string should define all layer groups."""
        pm = PotentialManager(mock_settings, use_langevin=False)
        pm.register_component('sheet', mos2_multilayer_config, n_layers=4)
        pm.add_self_interaction('sheet')
        
        layer_groups = pm.get_layer_groups_string()
        
        assert 'layer_1' in layer_groups
        assert 'layer_2' in layer_groups
        assert 'layer_3' in layer_groups
        assert 'layer_4' in layer_groups


# =========================================================================
# get_single_component_commands Tests (for amorphisation)
# =========================================================================

class TestSingleComponentCommands:
    """Tests for get_single_component_commands method (used by make_amorphous)."""

    def test_get_commands_for_sw(self, mock_settings, si_tip_config):
        """Should generate valid LAMMPS commands for SW potential."""
        pm = PotentialManager(mock_settings, use_langevin=False)
        commands = pm.get_single_component_commands(si_tip_config, ['Si'])
        
        assert len(commands) >= 3  # pair_style, pair_coeff, mass
        assert any('pair_style' in c for c in commands)
        assert any('pair_coeff' in c for c in commands)
        assert any('mass' in c for c in commands)
        # SW should include itself
        assert any('sw' in c for c in commands)

    def test_get_commands_includes_all_elements(self, mock_settings, mos2_sheet_config):
        """Commands should include all elements."""
        pm = PotentialManager(mock_settings, use_langevin=False)
        commands = pm.get_single_component_commands(mos2_sheet_config, ['Mo', 'S'])
        
        # Should have pair_style, pair_coeff, and 2 mass commands
        mass_commands = [c for c in commands if 'mass' in c]
        assert len(mass_commands) == 2

    def test_get_commands_hybrid_for_lj_requiring(self, mock_settings, si_tip_config):
        """get_single_component_commands uses bare pair_style for single-component setup."""
        pm = PotentialManager(mock_settings, use_langevin=False)
        commands = pm.get_single_component_commands(si_tip_config, ['Si'])
        
        pair_style = [c for c in commands if 'pair_style' in c][0]
        # Single-component amorphisation commands use a bare pair_style (no LJ needed)
        assert 'sw' in pair_style
        assert pair_style.strip() == 'pair_style sw'


# =========================================================================
# File Output Tests
# =========================================================================

class TestFileOutput:
    """Tests for LAMMPS file generation."""

    def test_write_file_creates_directory(
        self, mock_settings, si_tip_config, mock_cifread_si, mock_count_atomtypes_si
    ):
        """write_file should create output directory if needed."""
        pm = PotentialManager(mock_settings, use_langevin=False)
        pm.register_component('tip', si_tip_config)
        pm.add_self_interaction('tip')
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'subdir' / 'system.in.settings'
            pm.write_file(output_path)
            
            assert output_path.exists()
            assert output_path.parent.exists()

    def test_write_file_contains_masses(
        self, mock_settings, si_tip_config, mock_cifread_si, mock_count_atomtypes_si
    ):
        """Output file should contain mass commands."""
        pm = PotentialManager(mock_settings, use_langevin=False)
        pm.register_component('tip', si_tip_config)
        pm.add_self_interaction('tip')
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'system.in.settings'
            pm.write_file(output_path)
            
            content = output_path.read_text()
            assert 'mass' in content
            assert 'Si' in content

    def test_write_file_pair_style(
        self, mock_settings, si_tip_config, mock_cifread_si, mock_count_atomtypes_si
    ):
        """Output file should contain pair_style command."""
        pm = PotentialManager(mock_settings, use_langevin=False)
        pm.register_component('tip', si_tip_config)
        pm.add_self_interaction('tip')
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'system.in.settings'
            pm.write_file(output_path)
            
            content = output_path.read_text()
            assert 'pair_style' in content
            assert 'sw' in content  # Si uses SW potential

    def test_write_file_layer_groups_only_for_multilayer(
        self, mock_settings, mos2_sheet_config, mock_cifread_mos2, mock_count_atomtypes_mos2_simple
    ):
        """Layer groups should only appear for multi-layer components."""
        # Single layer
        pm = PotentialManager(mock_settings, use_langevin=False)
        pm.register_component('sheet', mos2_sheet_config, n_layers=1)
        pm.add_self_interaction('sheet')
        
        layer_groups = pm.get_layer_groups_string()
        assert layer_groups == ''  # No layer groups for single layer

        # Multi-layer
        pm2 = PotentialManager(mock_settings, use_langevin=False)
        pm2.register_component('sheet', mos2_sheet_config, n_layers=4)
        pm2.add_self_interaction('sheet')
        
        layer_groups = pm2.get_layer_groups_string()
        assert 'layer_1' in layer_groups
        assert 'layer_4' in layer_groups

    def test_write_file_includes_sheet_edge_groups(
        self, mock_settings, mos2_sheet_config, mock_cifread_mos2, mock_count_atomtypes_mos2_simple
    ):
        """Sheet edge typing should emit core and edge groups."""
        pm = PotentialManager(mock_settings, use_langevin=False)
        pm.register_component('sheet', mos2_sheet_config, n_layers=1, regions=[None, 'edge'])
        pm.add_self_interaction('sheet')

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'system.in.settings'
            pm.write_file(output_path)

            content = output_path.read_text()
            assert 'group 2D_core type' in content
            assert 'group 2D_edge type' in content


# =========================================================================
# Gap Calculation Tests
# =========================================================================

class TestGapCalculation:
    """Tests for gap calculation between components."""

    def test_calculate_gap_same_elements(
        self, mock_settings, si_tip_config, si_sub_config, 
        mock_cifread_si, mock_count_atomtypes_si
    ):
        """Gap between same elements should use their sigma."""
        pm = PotentialManager(mock_settings, use_langevin=False)
        pm.register_component('tip', si_tip_config)
        pm.register_component('sub', si_sub_config)
        
        gap = pm.calculate_gap('tip', 'sub', buffer=0.5)
        
        # Si-Si sigma from UFF + buffer
        assert gap > 0.5  # At least the buffer
        assert gap < 10.0  # Reasonable upper bound

    def test_calculate_gap_different_elements(self, mock_settings, tmp_path):
        """Gap between different elements should use mixing rules."""
        import src.core.potential_manager as pm_module
        from src.core.config import SheetConfig, TipConfig
        
        def mock_cifread(path):
            if 'MoS2' in str(path):
                return {'elements': ['Mo', 'S']}
            return {'elements': ['Si']}
        
        def mock_count(path, elements, pot_type=None):
            return {el: 1 for el in elements}
        
        si_sw = tmp_path / 'Si.sw'
        mos2_sw = tmp_path / 'MoS2.sw'
        si_cif = tmp_path / 'Si.cif'
        mos2_cif = tmp_path / 'MoS2.cif'
        for p in [si_sw, mos2_sw, si_cif, mos2_cif]:
            p.write_text('# dummy', encoding='utf-8')
        
        tip_config = TipConfig(
            mat='Si', pot_type='sw', pot_path=str(si_sw),
            cif_path=str(si_cif), r=20.0, amorph='c', dspring=0.1,
        )
        sheet_config = SheetConfig(
            mat='MoS2', pot_type='sw', pot_path=str(mos2_sw),
            cif_path=str(mos2_cif), x=50.0, y=50.0, layers=[1]
        )
        
        original_cifread = pm_module.cifread
        original_count = pm_module.count_atomtypes
        pm_module.cifread = mock_cifread
        pm_module.count_atomtypes = mock_count
        
        try:
            pm = PotentialManager(mock_settings, use_langevin=False)
            pm.register_component('tip', tip_config)
            pm.register_component('sheet', sheet_config)
            
            gap = pm.calculate_gap('tip', 'sheet', buffer=0.5)
            
            # Should be positive and reasonable
            assert gap > 0.5
            assert gap < 10.0
        finally:
            pm_module.cifread = original_cifread
            pm_module.count_atomtypes = original_count


# =========================================================================
# Edge Cases and Error Handling
# =========================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_unregistered_component_raises_error(self, mock_settings):
        """Interlayer setup on unregistered component should raise ValueError."""
        pm = PotentialManager(mock_settings)
        
        with pytest.raises(ValueError, match="not registered"):
            pm.add_ghost_lj('nonexistent')

    def test_add_cross_interaction_missing_component(self, mock_settings):
        """Cross-interaction with missing component should raise error."""
        pm = PotentialManager(mock_settings)
        
        with pytest.raises(ValueError, match="not found"):
            pm.add_cross_interaction('comp1', 'comp2')

    def test_multiple_components_type_ids_sequential(
        self, mock_settings, si_tip_config, si_sub_config,
        mock_cifread_si, mock_count_atomtypes_si
    ):
        """Multiple components should get sequential type IDs."""
        pm = PotentialManager(mock_settings, use_langevin=False)
        
        tip_map = pm.register_component('tip', si_tip_config)
        sub_map = pm.register_component('sub', si_sub_config)
        
        # Tip gets type 1, sub gets type 2
        assert tip_map['Si'] == [1]
        assert sub_map['Si'] == [2]
        assert len(pm.types) == 2

    def test_group_string_unregistered_returns_empty(self, mock_settings):
        """Group string for unregistered component should be empty."""
        pm = PotentialManager(mock_settings)
        assert pm.types.get_group_string('nonexistent') == ''

    def test_add_self_interaction_before_registration(self, mock_settings):
        """Adding self-interaction before registration should raise error."""
        pm = PotentialManager(mock_settings)
        
        with pytest.raises(ValueError, match="not registered"):
            pm.add_self_interaction('nonexistent')


# =========================================================================
# MLIP emission (golden blocks: exact LAMMPS text)
# =========================================================================

class TestMLIPEmission:
    """Golden-block tests for the one MLIP topology the matrix allows.

    Every assertion is against a WHOLE emitted line (``line in text.splitlines()``),
    not a substring, because this is the text a LAMMPS parser sees verbatim.

    Type layout note: ``_assign_types`` is ELEMENT-first (all layers of Mo, then
    all layers of S), so a 3-layer MoS2 sheet is types 1,2,3 = Mo(L1..L3) and
    4,5,6 = S(L1..L3).

    The per-layer / NULL-partitioned goldens that used to live here are gone
    with v4: both frameworks reject NULL type maps (v4 spec §1, E2/E3), so a
    test pinning that text would be pinning output no LAMMPS build accepts.
    """

    def test_deepmd_multilayer_is_one_full_coverage_instance(
            self, mock_settings, tmp_path, mock_structure_readers):
        """3 layers -> ONE un-indexed deepmd, full element list, no NULL, no LJ."""
        model = _fake_model(tmp_path, "m.pth")
        pm = PotentialManager(mock_settings, use_langevin=False)
        cfg = make_component_config(tmp_path, pot_type="deepmd", pot_path=model)
        pm.register_component("sheet", cfg, n_layers=3)
        pm.add_self_interaction("sheet")
        text = write_and_read(pm, tmp_path)
        lines = text.splitlines()

        # A lone full-cell MLIP needs no hybrid wrapper at all.
        assert f"pair_style deepmd {model}" in lines
        assert "pair_coeff * * Mo Mo Mo S S S # sheet" in lines
        assert "NULL" not in text
        assert "deepmd 1" not in text
        assert "lj/cut" not in text
        # An MLIP is cell-scoped now (v6 §3): its line is materialised at write
        # time rather than stored per component, so assert on the deck itself.
        assert len([l for l in lines if l.startswith("pair_coeff")]) == 1
        # The model-deviation trap: never two model paths on one style token.
        assert f"deepmd {model} {model}" not in text

    def test_deepmd_multilayer_with_virtual_keeps_the_full_map(
            self, mock_settings, tmp_path, mock_structure_readers):
        """A virtual atom alone does NOT force `hybrid`; the map stays full.

        E3 (`pair_deepmd.cpp:816`) forbids restricting deepmd to a type range,
        so the coeff line stays ``pair_coeff * *`` with one element per type --
        the virtual type taking the placeholder element. Because that map
        already covers the virtual type (and `neigh_modify exclude` keeps it
        inert), no lj/cut placeholder is needed, so a lone MLIP emits a plain
        non-hybrid pair_style.
        """
        model = _fake_model(tmp_path, "m.pth")
        pm = PotentialManager(mock_settings, use_langevin=False)
        cfg = make_component_config(tmp_path, pot_type="deepmd", pot_path=model)
        pm.register_component("sheet", cfg, n_layers=3)
        v = pm.register_virtual_atom()
        pm.add_self_interaction("sheet")
        text = write_and_read(pm, tmp_path)
        lines = text.splitlines()

        assert v == 7
        assert f"pair_style deepmd {model}" in lines
        # Non-hybrid: the coeff line carries no sub-style selector.
        assert "pair_coeff * * Mo Mo Mo S S S Mo # sheet" in lines
        assert "NULL" not in text
        # The virtual type is covered by `* *`, never reassigned to lj/cut.
        assert "lj/cut" not in text
        assert f"pair_coeff * {v} lj/cut" not in text

    def test_virtual_atom_is_never_reassigned_away_from_an_mlip(
            self, mock_settings, tmp_path, mock_structure_readers):
        """Regression: a `pair_coeff * <v> lj/cut` line must not be emitted.

        Under plain ``hybrid`` each type pair belongs to exactly ONE sub-style,
        so a trailing ``pair_coeff * <v> lj/cut`` REASSIGNS the virtual type
        away from the MLIP -- leaving its type map declaring an element for a
        type it no longer owns. LAMMPS segfaults the moment ``create_atoms``
        makes a real virtual atom (i.e. at the sliding phase, after a full
        equilibration has already run). The MLIP map covers the virtual type
        itself and `neigh_modify exclude` keeps it inert, so no placeholder is
        needed at any point.
        """
        for dispersion in (False, True):
            model = _fake_model(tmp_path, f"m_{dispersion}.pth")
            pm = PotentialManager(mock_settings, use_langevin=False)
            cfg = make_component_config(tmp_path, pot_type="deepmd",
                                        pot_path=model)
            pm.register_component("sheet", cfg, n_layers=3)
            v = pm.register_virtual_atom()
            pm.add_self_interaction("sheet")
            if dispersion:
                pm.enable_d3()
            text = write_and_read(pm, tmp_path / f"d3_{dispersion}")

            assert f"pair_coeff * {v} lj/cut" not in text, (
                f"virtual type {v} reassigned away from the MLIP "
                f"(dispersion={dispersion})")
            assert "lj/cut" not in text, f"dispersion={dispersion}"
            # The MLIP still owns every type, virtual one included.
            assert "Mo Mo Mo S S S Mo" in text, f"dispersion={dispersion}"
            for t in range(1, 8):
                assert f"neigh_modify exclude type {v} {t}" in text

    def test_mace_fullcell_with_d3_and_virtual(self, mock_settings, tmp_path,
                                               mock_structure_readers):
        """Full-cell + D3 mechanism incl. virtual-atom placeholder + exclusion."""
        model = _fake_model(tmp_path, "m.json")
        pm = PotentialManager(mock_settings, use_langevin=False)
        cfg = make_component_config(tmp_path, pot_type="mace", pot_path=model)
        pm.register_component("sheet", cfg, n_layers=3)
        v = pm.register_virtual_atom()
        pm.add_self_interaction("sheet")
        pm.enable_d3()
        text = write_and_read(pm, tmp_path)
        lines = text.splitlines()

        assert v == 7
        assert ("pair_style hybrid/overlay symmetrix/mace "
                "dispersion/d3 bj pbe 16.0 8.0") in lines
        # One MACE instance spans every layer; the virtual type carries the
        # placeholder element because pair_mace rejects NULL.
        assert f"pair_coeff * * symmetrix/mace {model} Mo Mo Mo S S S Mo # sheet" in lines
        assert "pair_coeff * * dispersion/d3 Mo Mo Mo S S S Mo" in lines
        # `neigh_modify exclude type` has NO wildcard form: enumerate the pairs
        # (see scripts/milestone0/m0_4_virtual_exclude.in).
        for t in range(1, 8):
            assert f"neigh_modify exclude type 7 {t}" in lines
        assert "neigh_modify exclude type 7 *" not in text
        # No lj/cut placeholder: `pair_coeff * *` already spans the virtual
        # type for both the MLIP and the D3 overlay.
        assert "lj/cut" not in text

    def test_deepmd_native_fullcell_emission(self, mock_settings, tmp_path,
                                             mock_structure_readers):
        """Non-hybrid deepmd takes its type map with NO sub-style selector.

        (`pair_coeff * * Mo S ...` -- the form scripts/milestone0/m0_4 documents.)
        """
        model = _fake_model(tmp_path, "m.pth")
        pm = PotentialManager(mock_settings, use_langevin=False)
        cfg = make_component_config(tmp_path, pot_type="deepmd", pot_path=model)
        pm.register_component("sheet", cfg, n_layers=3)
        pm.add_self_interaction("sheet")
        text = write_and_read(pm, tmp_path)
        lines = text.splitlines()

        assert f"pair_style deepmd {model}" in lines
        assert "pair_coeff * * Mo Mo Mo S S S # sheet" in lines
        # Interlayer coupling comes from the model: no LJ anywhere.
        assert "lj/cut" not in text
        # Cell-scoped emission (v6 §3): assert on the deck, not on internal state.
        assert len([l for l in lines if l.startswith("pair_coeff")]) == 1


# =========================================================================
# Shared MLIP model staging
# =========================================================================

class TestSharedModelStaging:
    """MLIP models stage once at the simulation root, not per run directory."""

    def test_mlip_model_staged_once_across_managers(
        self, mock_settings, tmp_path, mock_structure_readers
    ):
        """Two runs sharing a root copy the model once; SW stays per-run.

        The MLIP and the SW component are deliberately in SEPARATE cells. They
        used to share one, but a cell mixing an MLIP with a classical potential
        is now refused at emission (v6 §4) -- an MLIP's ``pair_coeff * *``
        claims every atom type, so the SW component's types would be silently
        absorbed. Staging is per-manager regardless, which is what this covers.
        """
        shared = tmp_path / 'potentials'
        model = _fake_model(tmp_path, 'm.pth')
        sw = tmp_path / 'Si.sw'
        sw.write_text('# dummy sw\n', encoding='utf-8')

        emitted = {}
        run_dirs = []
        for run in ('run1', 'run2'):
            run_pots = tmp_path / run / 'provenance' / 'potentials'

            def _pm():
                return PotentialManager(
                    mock_settings, use_langevin=False,
                    potentials_dir=run_pots,
                    potentials_prefix=f'{run}/provenance/potentials',
                    shared_potentials_dir=shared,
                    shared_potentials_prefix='../potentials',
                )

            pm_mlip = _pm()
            pm_mlip.register_component(
                'sheet', make_component_config(tmp_path, 'deepmd', model),
                n_layers=1,
            )
            pm_mlip.add_self_interaction('sheet')

            pm_sw = _pm()
            pm_sw.register_component(
                'tip', make_component_config(tmp_path, 'sw', str(sw), kind='tip')
            )
            pm_sw.add_self_interaction('tip')

            emitted[run] = (
                write_and_read(pm_mlip, tmp_path / run, 'mlip.settings'),
                write_and_read(pm_sw, tmp_path / run, 'sw.settings'),
            )
            run_dirs.append(run_pots)

        # Exactly one copy of the model, at the shared root.
        assert sorted(p.name for p in shared.iterdir()) == ['m.pth']
        for run_pots in run_dirs:
            assert not (run_pots / 'm.pth').exists()
            # Classical potentials keep today's per-run staging.
            assert (run_pots / 'Si.sw').exists()

        for run, (mlip_text, sw_text) in emitted.items():
            # A lone full-cell MLIP needs no hybrid wrapper.
            assert 'pair_style deepmd ../potentials/m.pth' in mlip_text
            assert f'{run}/provenance/potentials/Si.sw' in sw_text

    def test_shared_model_not_recopied(
        self, mock_settings, tmp_path, mock_structure_readers
    ):
        """The second manager reuses the staged file rather than copying again."""
        shared = tmp_path / 'potentials'
        model = _fake_model(tmp_path, 'm.pth')

        def stage():
            pm = PotentialManager(
                mock_settings, use_langevin=False,
                shared_potentials_dir=shared,
                shared_potentials_prefix='../potentials',
            )
            return pm._get_potential_path(model)

        assert stage() == '../potentials/m.pth'
        first = (shared / 'm.pth').stat().st_mtime_ns
        assert stage() == '../potentials/m.pth'
        assert (shared / 'm.pth').stat().st_mtime_ns == first


# =========================================================================
# v6: cell-scoped MLIP type maps
# =========================================================================

class TestBuildFullMap:
    """``build_full_map`` spans every atom type in the cell, with no NULL.

    This is the map a full-cell MLIP needs (v6 spec §3): both frameworks
    reject NULL entries (v4 E2/E3), so a multi-component MLIP cell must be
    described by ONE ``pair_coeff * *`` naming a real element for every type.
    ``build_null_map`` stays as it is -- classical potentials still partition.
    """

    def test_full_map_spans_every_component_with_no_null(
            self, mock_settings, tmp_path, mock_structure_readers):
        """sub(Si x3) + tip(Si x3) + sheet(Mo,S) -> 8 entries, in type-id order."""
        model = _fake_model(tmp_path, 'm.pth')
        pm = PotentialManager(mock_settings, use_langevin=True)
        pm.register_component('sub', make_sub_config(tmp_path, 'deepmd', model))
        pm.register_component(
            'tip', make_component_config(tmp_path, 'deepmd', model, kind='tip'))
        pm.register_component(
            'sheet', make_component_config(tmp_path, 'deepmd', model))

        full = pm.types.build_full_map()

        assert full == ['Si', 'Si', 'Si', 'Si', 'Si', 'Si', 'Mo', 'S']
        assert 'NULL' not in full
        assert len(full) == len(pm.types)

    def test_full_map_matches_null_map_for_a_single_component_cell(
            self, mock_settings, tmp_path, mock_structure_readers):
        """With one component there is nothing to pad, so the two agree."""
        model = _fake_model(tmp_path, 'm.pth')
        pm = PotentialManager(mock_settings, use_langevin=False)
        pm.register_component(
            'sheet', make_component_config(tmp_path, 'deepmd', model), n_layers=3)

        assert pm.types.build_full_map() == pm.types.build_null_map('sheet')

    def test_null_map_still_partitions(
            self, mock_settings, tmp_path, mock_structure_readers):
        """Regression: classical partitioning behaviour is untouched."""
        model = _fake_model(tmp_path, 'm.pth')
        pm = PotentialManager(mock_settings, use_langevin=False)
        pm.register_component(
            'tip', make_component_config(tmp_path, 'deepmd', model, kind='tip'))
        pm.register_component(
            'sheet', make_component_config(tmp_path, 'deepmd', model))

        assert pm.types.build_null_map('tip') == ['Si', 'NULL', 'NULL']
        assert pm.types.build_null_map('sheet') == ['NULL', 'Mo', 'S']


class TestOneMLIPInstancePerCell:
    """An MLIP is scoped to the CELL, not to each component (v6 spec §3).

    ``potential_usage`` drives how many sub-style tokens ``write_file`` emits.
    One MLIP instance must cover the whole cell, so registering three MLIP
    components must still count as one -- consistent with rule 1, which already
    forbids two different models of one framework in a cell.
    """

    def test_three_mlip_components_count_as_one_instance(
            self, mock_settings, tmp_path, mock_structure_readers):
        model = _fake_model(tmp_path, 'm.pth')
        pm = PotentialManager(mock_settings, use_langevin=False)
        for name, cfg in (
            ('sub', make_sub_config(tmp_path, 'deepmd', model)),
            ('tip', make_component_config(tmp_path, 'deepmd', model, kind='tip')),
            ('sheet', make_component_config(tmp_path, 'deepmd', model)),
        ):
            pm.register_component(name, cfg)

        assert pm.potential_usage['deepmd'] == 1

    def test_multilayer_mlip_sheet_still_counts_as_one(
            self, mock_settings, tmp_path, mock_structure_readers):
        """is_sheet_lj is False for an MLIP, so layers never multiplied it."""
        model = _fake_model(tmp_path, 'm.pth')
        pm = PotentialManager(mock_settings, use_langevin=False)
        pm.register_component(
            'sheet', make_component_config(tmp_path, 'deepmd', model), n_layers=3)

        assert pm.potential_usage['deepmd'] == 1

    def test_classical_usage_counts_are_unchanged(
            self, mock_settings, tmp_path, mock_structure_readers):
        """Regression: SW still counts per component, and per layer for a sheet."""
        sw = tmp_path / 'Si.sw'
        sw.write_text('# dummy sw\n', encoding='utf-8')
        pm = PotentialManager(mock_settings, use_langevin=False)
        pm.register_component(
            'tip', make_component_config(tmp_path, 'sw', str(sw), kind='tip'))
        pm.register_component(
            'sheet', make_component_config(tmp_path, 'sw', str(sw)), n_layers=3)

        # 1 for the tip + 3 for the 3-layer sheet (is_sheet_lj('sw') is True).
        assert pm.potential_usage['sw'] == 4


class TestCellScopedMLIPEmission:
    """One ``pair_coeff * *`` spanning every component (v6 spec §3/§4).

    Emission is DEFERRED to ``write_file``: the full map must span components
    registered *after* the first ``add_self_interaction`` call, which is the
    order ``hetero_afm`` and ``flake_slide`` use.
    """

    def _afm_cell(self, mock_settings, tmp_path, pot='deepmd', tip_pot=None):
        model = _fake_model(tmp_path, 'm.pth')
        pm = PotentialManager(mock_settings, use_langevin=False)
        pm.register_component('sub', make_sub_config(tmp_path, pot, model))
        tip_path = model
        if tip_pot is not None and tip_pot != pot:
            tip_path = str(tmp_path / 'Si.sw')
            Path(tip_path).write_text('# dummy sw\n', encoding='utf-8')
        pm.register_component(
            'tip', make_component_config(
                tmp_path, tip_pot or pot, tip_path, kind='tip'))
        pm.register_component('sheet', make_component_config(tmp_path, pot, model))
        return pm, model

    def test_afm_cell_emits_one_full_map_coeff_line(
            self, mock_settings, tmp_path, mock_structure_readers):
        """3 MLIP components -> ONE un-indexed pair_coeff over all 4 types."""
        pm, model = self._afm_cell(mock_settings, tmp_path)
        for name in ('sub', 'tip', 'sheet'):
            pm.add_self_interaction(name)
        text = write_and_read(pm, tmp_path)
        lines = text.splitlines()

        assert f'pair_style deepmd {model}' in lines
        assert len([l for l in lines if l.startswith('pair_coeff')]) == 1
        assert 'pair_coeff * * Si Si Mo S # sub, tip, sheet' in lines
        assert 'NULL' not in text
        assert 'deepmd 1' not in text
        assert 'lj/cut' not in text

    def test_component_registered_after_self_interaction_is_in_the_map(
            self, mock_settings, tmp_path, mock_structure_readers):
        """The ordering hazard deferral exists to remove (hetero_afm/flake)."""
        model = _fake_model(tmp_path, 'm.pth')
        pm = PotentialManager(mock_settings, use_langevin=False)
        pm.register_component('sheet', make_component_config(tmp_path, 'deepmd', model))
        pm.add_self_interaction('sheet')
        # Registered LATE, after the sheet's self-interaction call.
        pm.register_component(
            'tip', make_component_config(tmp_path, 'deepmd', model, kind='tip'))
        pm.add_self_interaction('tip')

        assert 'pair_coeff * * Mo S Si # sheet, tip' in write_and_read(
            pm, tmp_path).splitlines()

    def test_mlip_plus_classical_cell_fails_loudly(
            self, mock_settings, tmp_path, mock_structure_readers):
        """An SW tip beside an MLIP sheet must never reach LAMMPS (v6 §4)."""
        pm, _ = self._afm_cell(mock_settings, tmp_path, tip_pot='sw')
        for name in ('sub', 'sheet'):
            pm.add_self_interaction(name)
        pm.add_self_interaction('tip')

        with pytest.raises(ValueError) as exc:
            write_and_read(pm, tmp_path)
        msg = str(exc.value)
        assert 'tip' in msg and 'sw' in msg and 'deepmd' in msg

    def test_two_mlip_frameworks_in_one_cell_fail_loudly(
            self, mock_settings, tmp_path, mock_structure_readers):
        """Each framework would need its own `pair_coeff * *`; the second wins."""
        deep = _fake_model(tmp_path, 'm.pth')
        mace = _fake_model(tmp_path, 'm.json')
        pm = PotentialManager(mock_settings, use_langevin=False)
        pm.register_component('sheet', make_component_config(tmp_path, 'deepmd', deep))
        pm.register_component(
            'tip', make_component_config(tmp_path, 'mace', mace, kind='tip'))
        pm.add_self_interaction('sheet')
        pm.add_self_interaction('tip')

        with pytest.raises(ValueError) as exc:
            write_and_read(pm, tmp_path)
        assert 'deepmd' in str(exc.value) and 'mace' in str(exc.value)

    def test_single_component_mlip_deck_is_unchanged(
            self, mock_settings, tmp_path, mock_structure_readers):
        """Regression against the existing v4 golden."""
        model = _fake_model(tmp_path, 'm.pth')
        pm = PotentialManager(mock_settings, use_langevin=False)
        pm.register_component(
            'sheet', make_component_config(tmp_path, 'deepmd', model), n_layers=3)
        pm.add_self_interaction('sheet')
        lines = write_and_read(pm, tmp_path).splitlines()

        assert f'pair_style deepmd {model}' in lines
        assert 'pair_coeff * * Mo Mo Mo S S S # sheet' in lines

    def test_write_file_is_idempotent(
            self, mock_settings, tmp_path, mock_structure_readers):
        """Materialising at write time must not accumulate lines."""
        pm, _ = self._afm_cell(mock_settings, tmp_path)
        for name in ('sub', 'tip', 'sheet'):
            pm.add_self_interaction(name)
        first = write_and_read(pm, tmp_path, 'a.settings')
        second = write_and_read(pm, tmp_path, 'b.settings')
        assert first == second


class TestMLIPSuppressesCrossLJ:
    """No LJ between components an MLIP already covers (v6 spec §3).

    This is the physics argument of v4 §2 applied between components rather
    than between layers: the MLIP cutoff already spans the tip/sheet gap, so a
    UFF-mixed LJ parameterised to BE that interaction double-counts.
    """

    def _all_mlip_afm(self, mock_settings, tmp_path):
        model = _fake_model(tmp_path, 'm.pth')
        pm = PotentialManager(mock_settings, use_langevin=False)
        pm.register_component('sub', make_sub_config(tmp_path, 'deepmd', model))
        pm.register_component(
            'tip', make_component_config(tmp_path, 'deepmd', model, kind='tip'))
        pm.register_component('sheet', make_component_config(tmp_path, 'deepmd', model))
        for name in ('sub', 'tip', 'sheet'):
            pm.add_self_interaction(name)
        return pm

    def test_afm_cross_interactions_are_suppressed(
            self, mock_settings, tmp_path, mock_structure_readers):
        """The three pairs afm.py adds (sub-tip, sub-sheet, tip-sheet)."""
        pm = self._all_mlip_afm(mock_settings, tmp_path)
        pm.add_cross_interaction('sub', 'tip')
        pm.add_cross_interaction('sub', 'sheet')
        pm.add_cross_interaction('tip', 'sheet')

        assert pm.cross_interaction_commands == []
        assert 'lj/cut' not in write_and_read(pm, tmp_path)

    def test_interlayer_interaction_is_suppressed_for_an_mlip_sheet(
            self, mock_settings, tmp_path, mock_structure_readers):
        """afm.py:455 / sheetonsheet call this for multi-layer sheets."""
        model = _fake_model(tmp_path, 'm.pth')
        pm = PotentialManager(mock_settings, use_langevin=False)
        pm.register_component(
            'sheet', make_component_config(tmp_path, 'deepmd', model), n_layers=3)
        pm.add_self_interaction('sheet')
        pm.add_interlayer_interaction('sheet')

        assert pm.cross_interaction_commands == []
        assert 'lj/cut' not in write_and_read(pm, tmp_path)

    def test_ghost_lj_is_suppressed_for_an_mlip_sheet(
            self, mock_settings, tmp_path, mock_structure_readers):
        model = _fake_model(tmp_path, 'm.pth')
        pm = PotentialManager(mock_settings, use_langevin=False)
        pm.register_component(
            'sheet', make_component_config(tmp_path, 'deepmd', model), n_layers=3)
        pm.add_self_interaction('sheet')
        pm.add_ghost_lj('sheet', max_real_distance=1)

        assert pm.cross_interaction_commands == []

    def test_cross_interaction_between_mlip_and_classical_raises(
            self, mock_settings, tmp_path, mock_structure_readers):
        """rule 2 should have caught this upstream -- never emit it silently."""
        model = _fake_model(tmp_path, 'm.pth')
        sw = tmp_path / 'Si.sw'
        sw.write_text('# dummy sw\n', encoding='utf-8')
        pm = PotentialManager(mock_settings, use_langevin=False)
        pm.register_component('sheet', make_component_config(tmp_path, 'deepmd', model))
        pm.register_component(
            'tip', make_component_config(tmp_path, 'sw', str(sw), kind='tip'))

        with pytest.raises(ValueError) as exc:
            pm.add_cross_interaction('tip', 'sheet')
        assert 'deepmd' in str(exc.value)

    def test_classical_cross_interactions_are_unchanged(
            self, mock_settings, tmp_path, mock_structure_readers):
        """Regression: an all-SW cell still gets its LJ cross terms."""
        sw = tmp_path / 'Si.sw'
        sw.write_text('# dummy sw\n', encoding='utf-8')
        pm = PotentialManager(mock_settings, use_langevin=False)
        pm.register_component(
            'tip', make_component_config(tmp_path, 'sw', str(sw), kind='tip'))
        pm.register_component('sheet', make_component_config(tmp_path, 'sw', str(sw)))
        pm.add_cross_interaction('tip', 'sheet')

        assert len(pm.cross_interaction_commands) == 2  # Si-Mo, Si-S
        assert 'lj/cut' in write_and_read(pm, tmp_path)


class TestMLIPElementCoverage:
    """The emitted map must not name an element the model lacks (v6 §4 rule 3b).

    This check earns its keep because the failure it prevents is SILENT: v4 E7
    showed that naming an element the model does not have does not error --
    LAMMPS maps the types to the model's first N elements and returns a
    plausible, wrong energy (-61.89 eV instead of -256.90 on the reference
    trilayer).
    """

    @staticmethod
    def _mace_json(tmp_path, atomic_numbers, name='m.json'):
        import json
        p = tmp_path / name
        p.write_text(json.dumps({'atomic_numbers': atomic_numbers,
                                 'num_elements': len(atomic_numbers)}),
                     encoding='utf-8')
        return str(p)

    def test_si_tip_on_a_mo_s_only_model_is_refused(
            self, mock_settings, tmp_path, mock_structure_readers):
        """The real mace-mpa-0-medium-Mo-S.json is exactly this case."""
        model = self._mace_json(tmp_path, [16, 42])  # S, Mo -- no Si
        pm = PotentialManager(mock_settings, use_langevin=False)
        pm.register_component('sheet', make_component_config(tmp_path, 'mace', model))
        pm.register_component(
            'tip', make_component_config(tmp_path, 'mace', model, kind='tip'))
        pm.add_self_interaction('sheet')
        pm.add_self_interaction('tip')

        with pytest.raises(ValueError) as exc:
            write_and_read(pm, tmp_path)
        msg = str(exc.value)
        assert 'Si' in msg
        assert 'chemical-symbols' in msg  # names the re-extraction fix

    def test_a_model_covering_every_element_passes(
            self, mock_settings, tmp_path, mock_structure_readers):
        model = self._mace_json(tmp_path, [14, 16, 42])  # Si, S, Mo
        pm = PotentialManager(mock_settings, use_langevin=False)
        pm.register_component('sheet', make_component_config(tmp_path, 'mace', model))
        pm.register_component(
            'tip', make_component_config(tmp_path, 'mace', model, kind='tip'))
        pm.add_self_interaction('sheet')
        pm.add_self_interaction('tip')

        assert 'Mo S Si' in write_and_read(pm, tmp_path)

    def test_deepmd_coverage_is_not_enforced_at_generation_time(
            self, mock_settings, tmp_path, mock_structure_readers):
        """type_map needs torch to read; v4 §3.3 forbids that dependency here."""
        model = _fake_model(tmp_path, 'm.pth')
        pm = PotentialManager(mock_settings, use_langevin=False)
        pm.register_component('sheet', make_component_config(tmp_path, 'deepmd', model))
        pm.register_component(
            'tip', make_component_config(tmp_path, 'deepmd', model, kind='tip'))
        pm.add_self_interaction('sheet')
        pm.add_self_interaction('tip')

        assert 'pair_coeff * * Mo S Si # sheet, tip' in write_and_read(
            pm, tmp_path).splitlines()


class TestSingleComponentCommandsRefuseMLIP:
    """`get_single_component_commands` must never emit an MLIP deck (v6 §5).

    Its output is `pair_style <pot>` + `pair_coeff * * <path> <elements>`,
    which is wrong for BOTH frameworks: DeePMD carries the model in the
    pair_style line, MACE in the pair_coeff line. Its only callers are the
    generation-time melt-quench builds, which cannot run an MLIP anyway.
    """

    def test_mlip_config_raises(self, mock_settings, tmp_path,
                                mock_structure_readers):
        model = _fake_model(tmp_path, 'm.pth')
        pm = PotentialManager(mock_settings, use_langevin=False)
        cfg = make_component_config(tmp_path, 'deepmd', model, kind='tip')
        with pytest.raises(ValueError) as exc:
            pm.get_single_component_commands(cfg, ['Si'])
        assert 'amorph' in str(exc.value) or 'deepmd' in str(exc.value)

    def test_classical_config_is_unchanged(self, mock_settings, tmp_path,
                                           mock_structure_readers):
        sw = tmp_path / 'Si.sw'
        sw.write_text('# dummy sw\n', encoding='utf-8')
        pm = PotentialManager(mock_settings, use_langevin=False)
        cfg = make_component_config(tmp_path, 'sw', str(sw), kind='tip')
        cmds = pm.get_single_component_commands(cfg, ['Si'])
        assert cmds[0] == 'pair_style sw'
        assert cmds[1].startswith('pair_coeff * * ')


class TestNonHybridPathForBothFrameworks:
    """A multi-component MLIP cell can now take the NON-hybrid path (v6 §8).

    With one instance and no cross-LJ, `_is_single_potential` is true, so
    `write_file` emits a bare `pair_style` and routes every line through
    `_strip_non_hybrid_prefix`. Before v6 only single-component cells reached
    that path, and the two frameworks need opposite treatment there: deepmd's
    model path lives in the pair_style line (so the coeff line loses its style
    token), MACE's lives in the coeff line (so it must survive).
    """

    def _cell(self, mock_settings, tmp_path, pot, model):
        pm = PotentialManager(mock_settings, use_langevin=False)
        pm.register_component('sheet', make_component_config(tmp_path, pot, model))
        pm.register_component(
            'tip', make_component_config(tmp_path, pot, model, kind='tip'))
        pm.add_self_interaction('sheet')
        pm.add_self_interaction('tip')
        return write_and_read(pm, tmp_path).splitlines()

    def test_deepmd_multi_component_non_hybrid(
            self, mock_settings, tmp_path, mock_structure_readers):
        model = _fake_model(tmp_path, 'm.pth')
        lines = self._cell(mock_settings, tmp_path, 'deepmd', model)

        assert f'pair_style deepmd {model}' in lines
        assert 'hybrid' not in '\n'.join(lines)
        # Style token stripped; the model path stays in pair_style only.
        assert 'pair_coeff * * Mo S Si # sheet, tip' in lines

    def test_mace_multi_component_non_hybrid_keeps_the_model_path(
            self, mock_settings, tmp_path, mock_structure_readers):
        """MACE carries its model in the COEFF line, so it must survive."""
        model = _fake_model(tmp_path, 'm.json')
        lines = self._cell(mock_settings, tmp_path, 'mace', model)

        style = [l for l in lines if l.startswith('pair_style')]
        coeff = [l for l in lines if l.startswith('pair_coeff')]
        assert len(style) == 1 and len(coeff) == 1
        assert 'hybrid' not in style[0]
        assert model in coeff[0], (
            'MACE model path must survive the non-hybrid rewrite')
        assert coeff[0].endswith('Mo S Si # sheet, tip')
