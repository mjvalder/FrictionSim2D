"""The document the GUI edits.

Its central property is that what the GUI writes is what the CLI reads, so the
round-trip tests over the real example configs are the load-bearing ones.
Qt-free, like the document itself.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.core.config import parse_config
from src.gui.controllers.document import ConfigDocument, DocumentError

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = REPO_ROOT / 'examples'

#: (config file, simulation model)
EXAMPLE_MODELS = [
    ('afm_config.ini', 'afm'),
    ('afm_flake_config.ini', 'afm'),
    ('indent_config.ini', 'indent'),
    ('pes_tip_config.ini', 'pes_tip'),
    ('pes_sheet_config.ini', 'pes_sheet'),
    ('sheet_config.ini', 'sheetonsheet'),
    ('afm_freestanding_pes_config.ini', 'pes_tip'),
    ('indent_campaign.ini', 'indent'),
    ('afm_freestanding_pes_campaign.ini', 'pes_tip'),
]

ALL_MODELS = ['afm', 'indent', 'pes_tip', 'sheetonsheet', 'pes_sheet']


class TestRoundTrip:
    """What the GUI writes must be what the CLI reads."""

    @pytest.mark.parametrize('name,model', EXAMPLE_MODELS, ids=[e[0] for e in EXAMPLE_MODELS])
    def test_load_and_save_preserves_every_value(self, name, model, tmp_path):
        document = ConfigDocument.from_file(EXAMPLES / name, model)
        out = document.save(tmp_path / name)
        assert parse_config(out) == parse_config(EXAMPLES / name)

    @pytest.mark.parametrize('name,model', EXAMPLE_MODELS, ids=[e[0] for e in EXAMPLE_MODELS])
    def test_a_second_pass_does_not_drift(self, name, model, tmp_path):
        first = ConfigDocument.from_file(EXAMPLES / name, model).save(tmp_path / 'a.ini')
        second = ConfigDocument.from_file(first, model).save(tmp_path / 'b.ini')
        assert parse_config(second) == parse_config(first)

    def test_saving_emits_readable_comments(self, tmp_path):
        document = ConfigDocument.from_file(EXAMPLES / 'indent_config.ini', 'indent')
        text = document.save(tmp_path / 'c.ini').read_text(encoding='utf-8')
        assert 'in Å' in text
        assert text.startswith('# FrictionSim2D')

    def test_comments_do_not_change_the_parsed_values(self, tmp_path):
        document = ConfigDocument.from_file(EXAMPLES / 'indent_config.ini', 'indent')
        out = document.save(tmp_path / 'c.ini')
        assert parse_config(out) == parse_config(EXAMPLES / 'indent_config.ini')


class TestHeteroGuard:
    """Numbered sheet sections route the build to the heterostructure builders,
    which this version does not describe."""

    def test_a_numbered_sheet_config_is_refused(self, tmp_path):
        config = tmp_path / 'hetero.ini'
        config.write_text(
            "[2D-1]\nmat = MoS2\n\n[2D-2]\nmat = WS2\n\n[general]\ntemp = 300\n",
            encoding='utf-8')
        with pytest.raises(DocumentError, match='heterostructure'):
            ConfigDocument.from_file(config, 'afm')

    def test_the_message_names_the_sections(self, tmp_path):
        config = tmp_path / 'hetero.ini'
        config.write_text("[2D-1]\nmat = MoS2\n\n[2D-2]\nmat = WS2\n", encoding='utf-8')
        with pytest.raises(DocumentError) as excinfo:
            ConfigDocument.from_file(config, 'afm')
        assert '2D-1' in str(excinfo.value) and '2D-2' in str(excinfo.value)


class TestBlankDocument:

    @pytest.mark.parametrize('model', ALL_MODELS)
    def test_every_required_section_is_present(self, model):
        document = ConfigDocument.blank(model)
        for section in document.sections():
            if not section.optional:
                assert document.has_section(section.key), section.key

    @pytest.mark.parametrize('model', ALL_MODELS)
    def test_optional_sections_start_absent(self, model):
        document = ConfigDocument.blank(model)
        for section in document.sections():
            if section.optional and not document.is_section_locked(section.key):
                assert not document.has_section(section.key), section.key

    def test_a_blank_document_is_not_dirty_until_edited(self):
        document = ConfigDocument.blank('afm')
        document.dirty = False
        document.set('general', 'temp', 500)
        assert document.dirty

    def test_blank_config_does_not_prefill_gated_fields(self):
        document = ConfigDocument.blank('afm')
        assert 'tip_x_offset' not in document.values['tip']


class TestSections:

    def test_optional_section_can_be_added_and_removed(self):
        document = ConfigDocument.blank('afm')
        assert not document.has_section('flake')
        document.enable_section('flake', True)
        assert document.has_section('flake')
        document.enable_section('flake', False)
        assert not document.has_section('flake')

    def test_substrate_cannot_be_removed_except_for_pes_tip(self):
        """Other models abort without one."""
        document = ConfigDocument.blank('afm')
        assert document.is_section_locked('sub')
        with pytest.raises(DocumentError, match='freestanding'):
            document.enable_section('sub', False)

    def test_pes_tip_may_be_freestanding(self):
        document = ConfigDocument.blank('pes_tip')
        assert not document.is_section_locked('sub')
        document.enable_section('sub', True)
        document.enable_section('sub', False)
        assert not document.has_section('sub')

    def test_unknown_section_raises(self):
        with pytest.raises(DocumentError, match='Unknown section'):
            ConfigDocument.blank('afm').enable_section('nope', True)

    def test_sheet_family_offers_no_tip(self):
        keys = [s.key for s in ConfigDocument.blank('sheetonsheet').sections()]
        assert 'tip' not in keys

    def test_mode_specific_sections_are_offered(self):
        assert 'indent' in [s.key for s in ConfigDocument.blank('indent').sections()]
        assert 'pes' in [s.key for s in ConfigDocument.blank('pes_tip').sections()]


class TestModelSwitching:

    def test_switching_keeps_shared_sections(self):
        document = ConfigDocument.blank('afm')
        document.set('2D', 'mat', 'h-MoS2')
        document.set_model('indent')
        assert document.get('2D', 'mat') == 'h-MoS2'

    def test_switching_drops_sections_the_new_model_cannot_use(self):
        document = ConfigDocument.blank('afm')
        document.enable_section('flake', True)
        document.set_model('sheetonsheet')
        assert not document.has_section('tip')
        assert not document.has_section('flake')

    def test_switching_adds_the_new_required_sections(self):
        document = ConfigDocument.blank('sheetonsheet')
        document.set_model('afm')
        assert document.has_section('tip')

    def test_unknown_model_raises(self):
        with pytest.raises(DocumentError, match='Unknown simulation model'):
            ConfigDocument.blank('afm').set_model('nonsense')


class TestValidation:

    def test_a_good_config_validates(self, monkeypatch):
        monkeypatch.chdir(REPO_ROOT)
        document = ConfigDocument.from_file(EXAMPLES / 'indent_config.ini', 'indent')
        assert document.validate().ok

    def test_relative_paths_resolve_against_the_config_not_the_cwd(self, tmp_path,
                                                                  monkeypatch):
        """The whole point of base_dir: a desktop launcher has an arbitrary cwd."""
        monkeypatch.chdir(tmp_path)
        document = ConfigDocument.from_file(EXAMPLES / 'indent_config.ini', 'indent')
        assert document.validate().ok

    def test_a_missing_file_is_reported_on_its_field(self, monkeypatch):
        monkeypatch.chdir(REPO_ROOT)
        document = ConfigDocument.from_file(EXAMPLES / 'indent_config.ini', 'indent')
        document.set('tip', 'pot_path', 'nope.sw')
        result = document.validate()
        assert not result.ok
        assert result.issues_for('tip', 'pot_path')

    def test_a_bad_type_is_reported_on_its_field(self, monkeypatch):
        monkeypatch.chdir(REPO_ROOT)
        document = ConfigDocument.from_file(EXAMPLES / 'indent_config.ini', 'indent')
        document.set('tip', 'r', 'enormous')
        result = document.validate()
        assert not result.ok
        assert result.issues_for('tip', 'r')


class TestExpansion:

    def test_counts_match_the_known_tree(self, monkeypatch):
        monkeypatch.chdir(REPO_ROOT)
        document = ConfigDocument.from_file(EXAMPLES / 'indent_config.ini', 'indent')
        summary = document.expansion()
        assert (summary.configs, summary.builds, summary.jobs) == (1, 2, 10)

    def test_a_temperature_list_multiplies_configs(self, monkeypatch):
        monkeypatch.chdir(REPO_ROOT)
        document = ConfigDocument.from_file(EXAMPLES / 'indent_config.ini', 'indent')
        document.set('general', 'temp', [300, 500])
        assert document.expansion().configs == 2


class TestExpansionOutputDirs:
    """The Generate confirmation previews these dirs and flags collisions
    among them -- the exact hazard the loopability rule exists to prevent."""

    def test_one_dir_per_config_matching_the_real_layout(self, monkeypatch):
        monkeypatch.chdir(REPO_ROOT)
        document = ConfigDocument.from_file(EXAMPLES / 'indent_config.ini', 'indent')
        dirs = document.expansion_output_dirs()
        assert len(dirs) == document.expansion().configs == 1
        assert str(dirs[0]) == 'indent/h-MoS2/60x_60y/sub_aSi_tip_Si_r25/K300'

    def test_a_sweep_gives_one_distinct_dir_per_member(self, monkeypatch):
        monkeypatch.chdir(REPO_ROOT)
        document = ConfigDocument.from_file(EXAMPLES / 'indent_config.ini', 'indent')
        document.set('general', 'temp', [300, 500])
        dirs = document.expansion_output_dirs()
        assert len(dirs) == 2
        assert len(set(dirs)) == 2

    def test_temperatures_truncating_to_the_same_kelvin_collide(self, monkeypatch):
        """K{int(temp)} truncates: a sweep of 300.1 and 300.9 both round down
        to K300 and would build into the very same directory, silently
        overwriting each other -- despite `temp` being a supported sweep
        field. This is exactly the collision the Generate confirmation must
        be able to surface."""
        monkeypatch.chdir(REPO_ROOT)
        document = ConfigDocument.from_file(EXAMPLES / 'indent_config.ini', 'indent')
        document.set('general', 'temp', [300.1, 300.9])
        dirs = document.expansion_output_dirs()
        assert len(dirs) == 2
        assert len(set(dirs)) == 1

    def test_a_config_that_cannot_expand_reports_no_dirs(self, monkeypatch):
        """expand_config_sweeps raises when a materials_list resolves to
        materials none of which are usable (see its docstring); that is a
        config mid-edit, not a crash, so this must swallow it rather than
        propagate -- exactly like the exception path expansion() itself
        already tolerates via _refresh_counts' try/except."""
        monkeypatch.chdir(REPO_ROOT)
        document = ConfigDocument.from_file(EXAMPLES / 'indent_config.ini', 'indent')
        monkeypatch.setattr(
            'src.gui.controllers.document.expand_config_sweeps',
            lambda *a, **k: (_ for _ in ()).throw(ValueError('no usable materials')))
        assert document.expansion_output_dirs() == []


class TestLoopability:
    """A list is only a sweep when it changes the output directory name."""

    @pytest.mark.parametrize('section,key,expected', [
        ('general', 'temp', 'sweep'),
        ('2D', 'layers', 'layer'),
        ('general', 'force', 'in_file'),
        ('general', 'scan_angle_force', 'in_file'),
        ('2D', 'x', 'scalar'),
        ('tip', 'r', 'scalar'),
    ])
    def test_classification(self, section, key, expected):
        assert ConfigDocument.blank('afm').loopability_of(section, key) == expected

    def test_a_list_in_an_unsweepable_field_is_reported(self):
        document = ConfigDocument.blank('afm')
        document.set('tip', 'r', [10, 20])
        assert ('tip', 'r') in document.unsupported_lists()

    def test_legitimate_lists_are_not_reported(self):
        document = ConfigDocument.blank('afm')
        document.set('general', 'force', [5, 10])
        document.set('2D', 'layers', [1, 2])
        assert document.unsupported_lists() == []


class TestDirtyTracking:

    def test_setting_the_same_value_does_not_dirty(self):
        document = ConfigDocument.blank('afm')
        document.set('general', 'temp', 300.0)
        document.dirty = False
        document.set('general', 'temp', 300.0)
        assert not document.dirty

    def test_saving_clears_dirty(self, tmp_path):
        document = ConfigDocument.blank('afm')
        document.set('general', 'temp', 500)
        document.save(tmp_path / 'c.ini')
        assert not document.dirty

    def test_saving_without_a_path_raises(self):
        with pytest.raises(DocumentError, match='No path'):
            ConfigDocument.blank('afm').save()
