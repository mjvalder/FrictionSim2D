"""Tests for MLIP support: binary-model short-circuits and the MLIP registry."""
import pytest

from frictionsim2d.core.mlip import (
    MLIP_REGISTRY,
    MLIPSpec,
    d3_coeff_line,
    d3_style_token,
    get_mlip,
    is_mlip,
)
from frictionsim2d.core.utils.structures import count_atomtypes


def test_count_atomtypes_skips_binary_mlip_models(tmp_path):
    model = tmp_path / "mos2.model-lammps.pt"
    model.write_bytes(b"\x80\x02\x00binary-not-utf8\xff")
    for pot_type in ("mace", "deepmd"):
        assert count_atomtypes(str(model), ["Mo", "S"], pot_type=pot_type) == {"Mo": 1, "S": 1}


def test_count_atomtypes_skips_by_extension_without_pot_type(tmp_path):
    for ext in (".pt", ".pth", ".pb"):
        model = tmp_path / f"m{ext}"
        model.write_bytes(b"\xff\xfe")
        assert count_atomtypes(str(model), ["Mo", "S"]) == {"Mo": 1, "S": 1}


# ---------------------------------------------------------------------------
# Registry (Task 4)
# ---------------------------------------------------------------------------

def test_is_mlip():
    assert is_mlip("mace") is True
    assert is_mlip("deepmd") is True
    assert is_mlip("MACE") is True          # case-insensitive
    assert is_mlip("sw") is False
    assert is_mlip("reaxff") is False
    assert is_mlip("") is False
    assert set(MLIP_REGISTRY) == {"mace", "deepmd"}
    assert all(isinstance(spec, MLIPSpec) for spec in MLIP_REGISTRY.values())


def test_style_token_per_framework():
    assert get_mlip("deepmd").style_token("m.pth") == "deepmd m.pth"
    # MACE is driven through symmetrix, whose style takes no extra keywords.
    assert get_mlip("mace").style_token("m.json") == "symmetrix/mace"


def test_coeff_line_mace_places_model_path():
    line = get_mlip("mace").coeff_line("", "m.json", ["Mo", "S", "NULL"])
    assert line == "pair_coeff * * symmetrix/mace m.json Mo S NULL"


def test_coeff_line_deepmd_places_index():
    line = get_mlip("deepmd").coeff_line(" 2", "m.pth", ["NULL", "Mo", "S"])
    assert line == "pair_coeff * * deepmd 2 NULL Mo S"


def test_registry_capability_flags():
    deepmd = get_mlip("deepmd")
    mace = get_mlip("mace")

    assert deepmd.model_in_style_line is True
    assert mace.model_in_style_line is False
    # v4 spec §1 E2/E3: pair_deepmd rejects NULL maps and type-range coeffs.
    assert deepmd.supports_null is False
    assert mace.supports_null is False          # ACEsuit #1314 -> full-cell only
    assert deepmd.supports_overlay is True
    assert mace.supports_overlay is True

    assert ".pth" in deepmd.file_extensions and ".pb" in deepmd.file_extensions
    # symmetrix consumes a .json extracted from the torch .model.
    assert mace.file_extensions == (".json",)
    assert mace.style_token_name == "symmetrix/mace"
    # pair_symmetrix_mace.cpp:172 / _kokkos.cpp:195
    assert any('atom_modify' in d for d in mace.deck_directives)
    assert any('newton' in d for d in mace.deck_directives)
    # newton-on forces neigh half; the two must stay paired.
    assert 'neigh half' in mace.launcher.lmp_flags

    assert deepmd.virtual_atom_element == "Mo"
    assert mace.virtual_atom_element == "Mo"

    assert deepmd.launcher.binary_key == "deepmd"
    assert deepmd.launcher.mpi_ranks == 1
    assert deepmd.launcher.lmp_flags == ""
    assert any("LAMMPS_PLUGIN_PATH" in line for line in deepmd.launcher.env_lines)

    assert mace.launcher.binary_key == "mace"
    assert mace.launcher.mpi_ranks == 1
    assert mace.launcher.lmp_flags == "-k on t ${OMP_NUM_THREADS:-8} -sf kk -pk kokkos neigh half"
    # REGRESSION GUARD -- the PACKAGE default must stay on the Kokkos host
    # backend. `-k on g 1` on a stock symmetrix returns -126.5489 eV where
    # every correct path returns -173.40779 eV (24-atom h-MoS2 bilayer,
    # measured 2026-08-31), with NO error raised: its `mace_types` view is
    # sized by model-element count but filled per atom type, so on CUDA only
    # 2 of 7 entries reach the device. That is fixed by
    # campaign_mace/symmetrix-mace_types-overflow.patch, but the fix is NOT
    # upstream, so we cannot assume it here. CUDA is a per-campaign opt-in
    # gated on the patch plus a reference-energy check; the host backend is
    # correct either way. See MLIPSpec.notes.
    assert " g " not in mace.launcher.lmp_flags
    assert "-k on t " in mace.launcher.lmp_flags
    assert any("OMP_NUM_THREADS" in line for line in mace.launcher.env_lines)

    # The registry documents the framework traps so callers cannot rediscover them.
    assert "model-deviation" in deepmd.notes or "model deviation" in deepmd.notes
    assert "NULL" in mace.notes


def test_deepmd_notes_record_the_executed_null_evidence():
    """The NULL refusal is measured, not inferred -- v4 spec §1, E2/E3/E7.

    Anyone tempted to restore partial-cell DeePMD has to read past this: the
    source line that raises, the exact hybrid error, and the silent-wrong-answer
    trap that makes the "just drop the element list" workaround worse than a
    crash.
    """
    notes = get_mlip("deepmd").notes

    # E3: the guard clause that refuses a type-range pair_coeff.
    assert "pair_deepmd.cpp:816" in notes
    # E2: the exact LAMMPS error a NULL-padded map produces.
    assert "Pair hybrid sub-style deepmd is not used" in notes
    # E7: omitting the element list does NOT error -- it silently mis-maps.
    assert "-61.89" in notes and "-256.90" in notes


def test_deepmd_env_line_covers_both_plugin_layouts():
    """conda and pip put the deepmd LAMMPS plugin in different directories.

    conda: ``$CONDA_PREFIX/lib/deepmd_lmp/dpplugin.so``
    pip:   ``<site-packages>/deepmd/lib/libdeepmd_lmpplugin.so``

    A line that resolves only one of them leaves LAMMPS reporting "Loaded 0
    plugins" and then "Unrecognized pair style 'deepmd'" on the other install.
    """
    lines = get_mlip("deepmd").launcher.env_lines
    joined = "\n".join(lines)

    assert any("LAMMPS_PLUGIN_PATH" in line for line in lines)
    assert "deepmd_lmp" in joined          # conda layout
    assert "deepmd/lib" in joined          # pip layout
    # There is no `deepmd.lmp` module in deepmd-kit v3; importing it raises and
    # the export silently becomes a traceback string.
    assert "deepmd.lmp" not in joined


def test_d3_lines(mock_settings):
    assert d3_style_token(mock_settings) == "dispersion/d3 bj pbe 16.0 8.0"
    assert (
        d3_coeff_line(["Mo", "S", "Mo", "S", "Mo"])
        == "pair_coeff * * dispersion/d3 Mo S Mo S Mo"
    )
    # d3 needs a real element for every type -- NULL is never valid here.
    assert "NULL" not in d3_coeff_line(["Mo", "S"])


def test_get_mlip_unknown_raises():
    with pytest.raises(ValueError) as exc:
        get_mlip("sw")
    message = str(exc.value)
    assert "sw" in message
    assert "mace" in message and "deepmd" in message


class TestFullCellMeansOneInstance:
    """Full-cell means ONE instance with a full type map -- v4 §3, v6 §4.

    Both frameworks reject NULL type maps (``pair_mace`` segfaults, ACEsuit
    #1314; ``pair_deepmd`` refuses outright), so a cell must be described by a
    single ``pair_coeff * *`` naming a real element for every type.

    v6 changed WHAT that permits. v4 §6 also refused a *uniform* multi-component
    cell, because the builders emitted one instance per component; they now emit
    one instance for the whole cell (``TypeRegistry.build_full_map``), so a
    tip+sheet+substrate cell on one framework is allowed. What stays refused is
    a cell that is not uniform: an MLIP beside a classical potential (rule 2),
    or two MLIP frameworks together (rule 3a).
    """

    @staticmethod
    def _sheet(tmp_path, pot_type, name, mat='h-MoS2'):
        from frictionsim2d.core.config import SheetConfig
        cif = tmp_path / 'x.cif'
        cif.write_text('#', encoding='utf-8')
        model = tmp_path / name
        model.write_bytes(b'\x00')
        # dispersion='d3': this is a 2-layer stack and a multilayer MLIP sheet
        # without D3 is rejected at config time (the bare model has no
        # interlayer binding -- see test_config.TestMultilayerMLIPRequiresD3).
        # These tests are about CELL topology, not dispersion.
        extra = {}
        if is_mlip(pot_type):
            # 2-layer stack: a multilayer MLIP sheet without D3 is rejected at
            # config time (the bare model has no interlayer binding -- see
            # test_config.TestMultilayerMLIPRequiresD3). These tests are about
            # CELL topology, not dispersion. Classical sheets must NOT get it:
            # `dispersion` is only valid on an MLIP pot_type.
            extra = {'dispersion': 'd3', 'lat_c': 6.15}
        return SheetConfig(mat=mat, pot_type=pot_type, pot_path=str(model),
                           cif_path=str(cif), x=30.0, y=30.0, layers=[2],
                           **extra)

    @staticmethod
    def _sw_tip(tmp_path):
        from frictionsim2d.core.config import TipConfig
        cif = tmp_path / 'Si.cif'
        cif.write_text('#', encoding='utf-8')
        sw = tmp_path / 'Si.sw'
        sw.write_text('# sw\n', encoding='utf-8')
        return TipConfig(mat='Si', pot_type='sw', pot_path=str(sw),
                         cif_path=str(cif), r=20.0, dspring=0.1)

    def test_two_mace_components_are_allowed(self, tmp_path):
        """v6 §4: one instance now spans every component, so this is buildable."""
        from frictionsim2d.core.mlip import validate_mlip_cell
        s1 = self._sheet(tmp_path, 'mace', 'm.json')
        s2 = self._sheet(tmp_path, 'mace', 'm.json', mat='h-WS2')
        validate_mlip_cell(s1, components=[('sheet_2', s2)])  # must not raise

    def test_single_mace_component_is_allowed(self, tmp_path):
        from frictionsim2d.core.mlip import validate_mlip_cell
        s1 = self._sheet(tmp_path, 'mace', 'm.json')
        validate_mlip_cell(s1, components=[])  # must not raise

    def test_single_deepmd_component_is_allowed(self, tmp_path):
        from frictionsim2d.core.mlip import validate_mlip_cell
        s1 = self._sheet(tmp_path, 'deepmd', 'm.pth')
        validate_mlip_cell(s1, components=[])  # must not raise

    def test_two_deepmd_components_are_allowed(self, tmp_path):
        """v6 §4: the map spans both components, so no NULL is needed."""
        from frictionsim2d.core.mlip import validate_mlip_cell
        s1 = self._sheet(tmp_path, 'deepmd', 'm.pth')
        s2 = self._sheet(tmp_path, 'deepmd', 'm.pth', mat='h-WS2')
        validate_mlip_cell(s1, components=[('sheet_2', s2)])  # must not raise

    def test_an_afm_shaped_all_deepmd_cell_is_allowed(self, tmp_path):
        """The whole point of v6: sheet + tip + substrate on one MLIP."""
        from frictionsim2d.core.config import SubstrateConfig, TipConfig
        from frictionsim2d.core.mlip import validate_mlip_cell
        cif = tmp_path / 'Si.cif'
        cif.write_text('#', encoding='utf-8')
        model = tmp_path / 'm.pth'
        model.write_bytes(b'\x00')
        sheet = self._sheet(tmp_path, 'deepmd', 'm.pth')
        tip = TipConfig(mat='Si', pot_type='deepmd', pot_path=str(model),
                        cif_path=str(cif), r=20.0, dspring=0.1)
        sub = SubstrateConfig(mat='Si', pot_type='deepmd', pot_path=str(model),
                              cif_path=str(cif), thickness=10.0)
        validate_mlip_cell(sheet, components=[('tip', tip), ('sub', sub)])

    def test_two_mlip_frameworks_in_one_cell_are_rejected(self, tmp_path):
        """rule 3a: each framework needs its own `* *`, so the second wins."""
        from frictionsim2d.core.config import TipConfig
        from frictionsim2d.core.mlip import validate_mlip_cell
        cif = tmp_path / 'Si.cif'
        cif.write_text('#', encoding='utf-8')
        mace_model = tmp_path / 'm.json'
        mace_model.write_bytes(b'\x00')
        sheet = self._sheet(tmp_path, 'deepmd', 'm.pth')
        tip = TipConfig(mat='Si', pot_type='mace', pot_path=str(mace_model),
                        cif_path=str(cif), r=20.0, dspring=0.1)
        with pytest.raises(ValueError) as exc:
            validate_mlip_cell(sheet, components=[('tip', tip)])
        msg = str(exc.value)
        assert 'deepmd' in msg and 'mace' in msg

    def test_deepmd_sheet_beside_an_sw_tip_is_rejected(self, tmp_path):
        """v4 §3: partial-cell is unavailable for DeePMD too (E2, E3)."""
        from frictionsim2d.core.mlip import validate_mlip_cell
        sheet = self._sheet(tmp_path, 'deepmd', 'm.pth')
        with pytest.raises(ValueError, match='(?i)partial-cell|classical'):
            validate_mlip_cell(sheet, components=[('tip', self._sw_tip(tmp_path))])

    def test_mace_sheet_beside_an_sw_tip_is_rejected(self, tmp_path):
        from frictionsim2d.core.mlip import validate_mlip_cell
        sheet = self._sheet(tmp_path, 'mace', 'm.json')
        with pytest.raises(ValueError, match='(?i)partial-cell|classical'):
            validate_mlip_cell(sheet, components=[('tip', self._sw_tip(tmp_path))])

    def test_deepmd_tip_beside_a_classical_sheet_is_rejected(self, tmp_path):
        """The mirror image: MLIP on the tip, classical sheet."""
        from frictionsim2d.core.config import TipConfig
        from frictionsim2d.core.mlip import validate_mlip_cell
        sheet = self._sheet(tmp_path, 'sw', 'MoS2_wen.sw')
        cif = tmp_path / 'Si.cif'
        cif.write_text('#', encoding='utf-8')
        model = tmp_path / 'm.pth'
        model.write_bytes(b'\x00')
        tip = TipConfig(mat='Si', pot_type='deepmd', pot_path=str(model),
                        cif_path=str(cif), r=20.0, dspring=0.1)
        with pytest.raises(ValueError, match='(?i)partial-cell|classical'):
            validate_mlip_cell(sheet, components=[('tip', tip)])

    def test_two_different_models_still_rejected_first(self, tmp_path):
        """The one-model-per-framework rule keeps its own diagnosis."""
        from frictionsim2d.core.mlip import validate_mlip_cell
        s1 = self._sheet(tmp_path, 'deepmd', 'a.pth')
        s2 = self._sheet(tmp_path, 'deepmd', 'b.pth', mat='h-WS2')
        with pytest.raises(ValueError, match='Two different deepmd model files'):
            validate_mlip_cell(s1, components=[('sheet_2', s2)])

    @pytest.mark.parametrize('mode', ['atom_bonds', 'com_spring'])
    def test_constraint_modes_still_rejected(self, tmp_path, mode):
        from frictionsim2d.core.mlip import validate_mlip_cell
        s1 = self._sheet(tmp_path, 'deepmd', 'm.pth')
        with pytest.raises(ValueError, match='constraint_mode'):
            validate_mlip_cell(s1, components=[], constraint_mode=mode)

    def test_classical_only_cell_is_untouched(self, tmp_path):
        from frictionsim2d.core.mlip import validate_mlip_cell
        sheet = self._sheet(tmp_path, 'sw', 'MoS2_wen.sw')
        validate_mlip_cell(sheet, components=[('tip', self._sw_tip(tmp_path))],
                           constraint_mode='atom_bonds')  # must not raise


class TestModelElements:
    """Reading a model's element set, for the coverage check (v6 §4 rule 3b)."""

    @staticmethod
    def _mace_json(tmp_path, atomic_numbers, name='m.json'):
        import json
        p = tmp_path / name
        p.write_text(json.dumps({
            'num_channels': 128, 'r_cut': 6.0,
            'atomic_numbers': atomic_numbers,
            'num_elements': len(atomic_numbers),
        }), encoding='utf-8')
        return str(p)

    def test_mace_elements_are_read_from_the_json(self, tmp_path):
        from frictionsim2d.core.mlip import model_elements
        # 16 = S, 42 = Mo -- the real mace-mpa-0-medium-Mo-S.json's contents.
        assert model_elements('mace', self._mace_json(tmp_path, [16, 42])) == {'S', 'Mo'}

    def test_mace_elements_include_si_after_re_extraction(self, tmp_path):
        from frictionsim2d.core.mlip import model_elements
        assert model_elements(
            'mace', self._mace_json(tmp_path, [14, 16, 42])) == {'Si', 'S', 'Mo'}

    def test_deepmd_element_set_is_unknown(self, tmp_path):
        """type_map lives inside a serialized .pth; reading it needs torch,
        which generation and CI must not require (v4 §3.3)."""
        from frictionsim2d.core.mlip import model_elements
        p = tmp_path / 'm.pth'
        p.write_bytes(b'\x00')
        assert model_elements('deepmd', str(p)) is None

    def test_unreadable_model_is_unknown_not_an_error(self, tmp_path):
        """A stub/binary/absent file must not break generation."""
        from frictionsim2d.core.mlip import model_elements
        stub = tmp_path / 'stub.json'
        stub.write_bytes(b'\x00')
        assert model_elements('mace', str(stub)) is None
        assert model_elements('mace', str(tmp_path / 'missing.json')) is None

    def test_only_the_file_prefix_is_read(self, tmp_path):
        """The real json is ~119 MB; the check must not load all of it."""
        import json
        p = tmp_path / 'big.json'
        head = json.dumps({'atomic_numbers': [14, 16, 42]})[:-1]
        p.write_text(head + ', "pad": "' + 'x' * 2_000_000 + '"}', encoding='utf-8')
        from frictionsim2d.core.mlip import model_elements
        assert model_elements('mace', str(p)) == {'Si', 'S', 'Mo'}


class TestAmorphousMLIPComponentsRefused:
    """v6 §5: `amorph='a'` under an MLIP is refused, not emitted.

    Amorphisation is force-dependent melt-quench MD run at GENERATION time
    (`components.py`), so it can use neither the `pair_style zero` surrogate
    (v4 §3.3 forbids it for runs that integrate) nor an MLIP, which would
    demand an MLIP-enabled LAMMPS on every generation machine and in CI.
    `get_single_component_commands` would emit a deck that is wrong for BOTH
    frameworks, so the cell is refused instead.
    """

    @staticmethod
    def _mlip_tip(tmp_path, amorph):
        from frictionsim2d.core.config import TipConfig
        cif = tmp_path / 'Si.cif'
        cif.write_text('#', encoding='utf-8')
        model = tmp_path / 'm.pth'
        model.write_bytes(b'\x00')
        return TipConfig(mat='Si', pot_type='deepmd', pot_path=str(model),
                         cif_path=str(cif), r=20.0, dspring=0.1, amorph=amorph)

    def test_amorphous_mlip_tip_is_refused(self, tmp_path):
        from frictionsim2d.core.mlip import validate_mlip_cell
        sheet = TestFullCellMeansOneInstance._sheet(tmp_path, 'deepmd', 'm.pth')
        with pytest.raises(ValueError) as exc:
            validate_mlip_cell(
                sheet, components=[('tip', self._mlip_tip(tmp_path, 'a'))])
        assert 'amorph' in str(exc.value)

    def test_crystalline_mlip_tip_is_allowed(self, tmp_path):
        from frictionsim2d.core.mlip import validate_mlip_cell
        sheet = TestFullCellMeansOneInstance._sheet(tmp_path, 'deepmd', 'm.pth')
        validate_mlip_cell(
            sheet, components=[('tip', self._mlip_tip(tmp_path, 'c'))])

    def test_amorphous_classical_tip_is_untouched(self, tmp_path):
        """Regression: this rule must only fire for MLIP cells."""
        from frictionsim2d.core.config import SheetConfig, TipConfig
        from frictionsim2d.core.mlip import validate_mlip_cell
        cif = tmp_path / 'x.cif'
        cif.write_text('#', encoding='utf-8')
        sw = tmp_path / 'Si.sw'
        sw.write_text('# sw\n', encoding='utf-8')
        sheet = SheetConfig(mat='h-MoS2', pot_type='sw', pot_path=str(sw),
                            cif_path=str(cif), x=30.0, y=30.0, layers=[2])
        tip = TipConfig(mat='Si', pot_type='sw', pot_path=str(sw),
                        cif_path=str(cif), r=20.0, dspring=0.1, amorph='a')
        validate_mlip_cell(sheet, components=[('tip', tip)])
