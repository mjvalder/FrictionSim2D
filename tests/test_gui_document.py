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
    ('hetero_pair_sweep.ini', 'sheetonsheet'),
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


PAIR_CONFIG = "[2D-1]\nmat = h_MoS2\n\n[2D-2]\nmat = h_WS2\n\n[general]\ntemp = 300\n"


def _config(tmp_path, text, name='hetero.ini'):
    path = tmp_path / name
    path.write_text(text, encoding='utf-8')
    return path


class TestSecondSheetLoading:
    """Numbered sheet sections route the build to the heterostructure
    builders. Every mode has one now (HeteroAFM/Indent/PESTip/PESSheet/
    SheetOnSheet), so a two-sheet config must be held, not refused -- but only
    the canonical `[2D-1]` + `[2D-2]` pair the document can represent.
    """

    @pytest.mark.parametrize('model', ALL_MODELS)
    def test_a_pair_is_accepted_by_every_mode(self, model, tmp_path):
        document = ConfigDocument.from_file(_config(tmp_path, PAIR_CONFIG), model)
        assert document.has_second_material()

    def test_two_sheet_config_round_trips(self, tmp_path):
        document = ConfigDocument.from_file(
            _config(tmp_path, PAIR_CONFIG), 'sheetonsheet')
        assert document.has_second_material()
        assert document.get('2D', 'mat') == 'h_MoS2'
        assert document.get('2D-2', 'mat') == 'h_WS2'
        out = document.normalised_values()
        assert {k for k in out if k.startswith('2D')} == {'2D-1', '2D-2'}

    def test_three_sheets_still_refused(self, tmp_path):
        config = _config(
            tmp_path, "[2D-1]\nmat = a\n\n[2D-2]\nmat = b\n\n[2D-3]\nmat = c\n")
        with pytest.raises(DocumentError, match='two'):
            ConfigDocument.from_file(config, 'sheetonsheet')

    def test_the_refusal_names_the_sections(self, tmp_path):
        config = _config(
            tmp_path, "[2D-1]\nmat = a\n\n[2D-2]\nmat = b\n\n[2D-3]\nmat = c\n")
        with pytest.raises(DocumentError) as excinfo:
            ConfigDocument.from_file(config, 'sheetonsheet')
        assert '2D-3' in str(excinfo.value)

    def test_plain_2d_mixed_with_numbered_sections_is_refused(self, tmp_path):
        """The model folds only the numbered sections into `sheets` and leaves
        `[2D]` to be overwritten by the first of them -- so a config carrying
        both means something different from what it looks like."""
        config = _config(tmp_path, "[2D]\nmat = a\n\n[2D-2]\nmat = b\n")
        with pytest.raises(DocumentError, match=r'\[2D\]'):
            ConfigDocument.from_file(config, 'sheetonsheet')

    def test_a_gap_in_the_numbering_is_refused(self, tmp_path):
        """`[2D-1]` + `[2D-3]` builds (the core folds by numeric order), but
        the document would have to silently renumber it to hold it."""
        config = _config(tmp_path, "[2D-1]\nmat = a\n\n[2D-3]\nmat = b\n")
        with pytest.raises(DocumentError, match='2D-2'):
            ConfigDocument.from_file(config, 'sheetonsheet')

    def test_a_lone_first_sheet_loads_as_a_single_material(self, tmp_path):
        config = _config(tmp_path, "[2D-1]\nmat = a\n\n[general]\ntemp = 300\n")
        document = ConfigDocument.from_file(config, 'sheetonsheet')
        assert not document.has_second_material()
        assert document.get('2D', 'mat') == 'a'
        assert '2D-1' not in document.normalised_values()


class TestSecondMaterial:
    """Holding, normalising and dropping the second sheet."""

    def test_single_material_emits_plain_2d(self):
        out = ConfigDocument.blank('sheetonsheet').normalised_values()
        assert '2D' in out and '2D-1' not in out

    def test_a_blank_document_has_no_second_material(self):
        assert not ConfigDocument.blank('sheetonsheet').has_second_material()

    def test_adding_one_defaults_its_fields(self):
        document = ConfigDocument.blank('sheetonsheet')
        document.add_second_material()
        assert document.has_second_material()
        assert document.get('2D-2', 'stack_type') == 'AA'

    def test_adding_twice_does_not_wipe_it(self):
        document = ConfigDocument.blank('sheetonsheet')
        document.add_second_material()
        document.set('2D-2', 'mat', 'h_WS2')
        document.add_second_material()
        assert document.get('2D-2', 'mat') == 'h_WS2'

    def test_normalised_values_renames_only_when_paired(self):
        document = ConfigDocument.blank('sheetonsheet')
        document.add_second_material()
        out = document.normalised_values()
        assert '2D' not in out
        assert list(out).index('2D-1') + 1 == list(out).index('2D-2')

    def test_normalised_values_is_a_copy(self):
        document = ConfigDocument.blank('sheetonsheet')
        document.normalised_values()['2D']['mat'] = 'mutated'
        assert document.get('2D', 'mat') != 'mutated'

    def test_the_second_sheet_spec_offers_no_sweep_field(self):
        """Only `[2D-1]` carries `materials_list`: the second member of each
        pair comes from the same line of the same list."""
        spec = ConfigDocument.blank('sheetonsheet').second_material_spec()
        assert spec.key == '2D-2'
        assert spec.optional
        keys = [f.key for f in spec.fields]
        assert 'materials_list' not in keys
        assert 'cif_path' in keys

    def test_layers_list_in_second_sheet_is_not_flagged(self):
        document = ConfigDocument.blank('sheetonsheet')
        document.add_second_material()
        document.set('2D-2', 'layers', [1])
        assert ('2D-2', 'layers') not in document.unsupported_lists()

    def test_a_scalar_list_in_the_second_sheet_is_still_flagged(self):
        document = ConfigDocument.blank('sheetonsheet')
        document.add_second_material()
        document.set('2D-2', 'x', [10, 20])
        assert ('2D-2', 'x') in document.unsupported_lists()

    def test_remove_second_material_pops_stacking(self):
        document = ConfigDocument.blank('sheetonsheet')
        document.add_second_material()
        document.set('general', 'hetero_stacking', 'alternating')
        document.remove_second_material()
        assert not document.has_second_material()
        assert document.get('general', 'hetero_stacking') is None

    def test_a_pair_saves_as_numbered_sections(self, tmp_path):
        document = ConfigDocument.blank('sheetonsheet')
        document.add_second_material()
        document.set('2D-2', 'mat', 'h_WS2')
        text = document.save(tmp_path / 'pair.ini').read_text(encoding='utf-8')
        assert '[2D-1]' in text and '[2D-2]' in text
        assert '\n[2D]\n' not in text

    def test_a_saved_pair_reopens_as_a_pair(self, tmp_path):
        document = ConfigDocument.blank('sheetonsheet')
        document.add_second_material()
        document.set('2D-2', 'mat', 'h_WS2')
        reopened = ConfigDocument.from_file(
            document.save(tmp_path / 'pair.ini'), 'sheetonsheet')
        assert reopened.get('2D-2', 'mat') == 'h_WS2'

    def test_the_written_pair_is_annotated_like_the_first_sheet(self, tmp_path):
        """`comments()` is keyed by section; without re-keying, a pair config
        would come out with every unit and help line stripped."""
        document = ConfigDocument.blank('sheetonsheet')
        document.add_second_material()
        annotations = document.comments()
        assert '2D-1' in annotations and '2D-2' in annotations
        assert '2D' not in annotations
        assert annotations['2D-2'] == annotations['2D-1']

    def test_the_second_sheets_own_override_annotates_it(self, monkeypatch):
        """One rule, two readers. The form reads `[2D-2]`'s own override entry
        and `comments()` used to fold `2D-2` onto `[2D]` instead; identical
        today only because the two entries carry the same text. An override
        added to the second sheet alone rendered on screen and was dropped on
        the way to the file."""
        from dataclasses import replace  # noqa: PLC0415
        from src.gui.schema import overrides as overrides_module  # noqa: PLC0415

        second = overrides_module._COMMON['2D-2']
        monkeypatch.setitem(
            overrides_module._COMMON, '2D-2',
            replace(second, fields={
                **second.fields,
                'mat': replace(second.fields['mat'],
                               help='the second material of the pair'),
            }))

        document = ConfigDocument.blank('sheetonsheet')
        document.add_second_material()
        annotations = document.comments()

        assert 'the second material of the pair' in annotations['2D-2']['mat']
        assert 'the second material of the pair' not in annotations['2D-1']['mat']

    def test_the_second_sheet_is_not_annotated_as_carrying_the_sweep_list(self, monkeypatch):
        """The other half of the same rule. `materials_list` is in `[2D]`'s
        override and deliberately not in `[2D-2]`'s -- one list names both
        materials of every pair -- so a hand-written one in the second section
        must not be commented as though the form offered it there."""
        from dataclasses import replace  # noqa: PLC0415
        from src.gui.schema import overrides as overrides_module  # noqa: PLC0415

        first = overrides_module._COMMON['2D']
        monkeypatch.setitem(
            overrides_module._COMMON, '2D',
            replace(first, fields={
                **first.fields,
                'materials_list': replace(first.fields['materials_list'],
                                          help='one line per material'),
            }))

        document = ConfigDocument.blank('sheetonsheet')
        document.add_second_material()
        document.set('2D', 'materials_list', 'pairs.txt')
        document.set('2D-2', 'materials_list', 'pairs.txt')
        annotations = document.comments()

        assert 'one line per material' in annotations['2D-1']['materials_list']
        assert 'materials_list' not in annotations['2D-2']

    def test_to_dict_carries_the_numbered_sections(self):
        document = ConfigDocument.blank('sheetonsheet')
        document.add_second_material()
        assert set(document.to_dict()) >= {'2D-1', '2D-2'}


class TestModelSwitchingWithASecondMaterial:

    @pytest.mark.parametrize('model', ALL_MODELS)
    def test_switching_keeps_material_2(self, model):
        document = ConfigDocument.blank('sheetonsheet')
        document.add_second_material()
        document.set('2D-2', 'mat', 'h_WS2')
        document.set_model(model)
        assert document.has_second_material()
        assert document.get('2D-2', 'mat') == 'h_WS2'

    def test_dropping_it_is_explicit(self):
        document = ConfigDocument.blank('sheetonsheet')
        document.add_second_material()
        document.set_model('afm', drop_second=True)
        assert not document.has_second_material()


class TestConflicts:

    def test_the_default_stacking_is_not_reported(self):
        """Every blank config carries `hetero_stacking = grouped`, because the
        model defaults it and `_defaults_for` seeds every defaulted key. That
        is not a setting anybody chose, and reporting it would put a conflict
        line on every single-material config the GUI ever opens."""
        document = ConfigDocument.blank('sheetonsheet')
        assert document.get('general', 'hetero_stacking') == 'grouped'
        assert document.conflicts() == []

    def test_stacking_without_a_second_material_is_reported(self):
        document = ConfigDocument.blank('sheetonsheet')
        document.set('general', 'hetero_stacking', 'alternating')
        assert any('hetero_stacking' in line for line in document.conflicts())

    def test_stacking_with_a_second_material_is_silent(self):
        document = ConfigDocument.blank('sheetonsheet')
        document.add_second_material()
        document.set('general', 'hetero_stacking', 'alternating')
        assert document.conflicts() == []

    def test_sweep_plus_second_material_is_not_a_conflict(self):
        """A material list and a second sheet together are the pair mode, not
        two features fighting."""
        document = ConfigDocument.blank('sheetonsheet')
        document.add_second_material()
        document.set('2D', 'materials_list', 'mats.txt')
        assert not any('mutually exclusive' in line
                       for line in document.conflicts())


class TestSecondMaterialValidation:

    def test_issues_map_to_form_sections(self, monkeypatch):
        monkeypatch.chdir(REPO_ROOT)
        document = ConfigDocument.blank('sheetonsheet')
        document.add_second_material()      # required fields empty -> errors
        result = document.validate()
        assert not result.ok
        sections = {i.section for i in result.issues}
        assert '2D-1' not in sections and 'sheets' not in sections

    def test_an_issue_keeps_the_field_that_caused_it(self, monkeypatch):
        """The pydantic loc is ('sheets', 1, 'cif_path'); mapping only the
        section leaves the issue keyed by the list index, which decorates no
        editor at all (SectionForm.show_errors matches on issue.key)."""
        monkeypatch.chdir(REPO_ROOT)
        document = ConfigDocument.blank('sheetonsheet')
        document.add_second_material()
        result = document.validate()
        assert result.issues_for('2D-2', 'cif_path'), result.render()

    def test_the_first_sheet_is_not_reported_twice(self, monkeypatch):
        """The model mirrors [2D-1] onto its `2D` field as well as into
        `sheets[0]`, so the same failure arrives twice."""
        monkeypatch.chdir(REPO_ROOT)
        document = ConfigDocument.blank('sheetonsheet')
        document.add_second_material()
        issues = document.validate().issues_for('2D', 'cif_path')
        assert len(issues) == 1, [i.render() for i in issues]

    def test_a_pair_sweep_validates_with_only_pending_templates(self, monkeypatch):
        """{mat1}/{mat2} are not errors when a pair list is configured."""
        monkeypatch.chdir(REPO_ROOT)
        document = ConfigDocument.from_file(
            EXAMPLES / 'hetero_pair_sweep.ini', 'sheetonsheet')
        result = document.validate()
        assert result.ok, result.render()
        assert document.pending_template_issues


class TestPairExpansion:

    def test_one_config_per_ordered_pair(self, monkeypatch):
        monkeypatch.chdir(REPO_ROOT)
        document = ConfigDocument.from_file(
            EXAMPLES / 'hetero_pair_sweep.ini', 'sheetonsheet')
        assert document.expansion().configs == 3

    def test_each_pair_gets_its_own_directory(self, monkeypatch):
        monkeypatch.chdir(REPO_ROOT)
        document = ConfigDocument.from_file(
            EXAMPLES / 'hetero_pair_sweep.ini', 'sheetonsheet')
        dirs = document.expansion_output_dirs().dirs
        assert len(dirs) == 3 and len(set(dirs)) == 3

    def test_a_wrongly_templated_pair_reports_its_guard_message(self, monkeypatch):
        """A [2D-2] left on '{mat1}' expands into a directory named after the
        pair while building the first material twice, so `expand_config_sweeps`
        refuses it with a sentence saying exactly that. The Generate preview
        must carry that sentence out: an empty directory list on its own reads
        as "this config generates nothing", which is a different problem with a
        different fix."""
        monkeypatch.chdir(REPO_ROOT)
        document = ConfigDocument.from_file(
            EXAMPLES / 'hetero_pair_sweep.ini', 'sheetonsheet')
        document.set('2D-2', 'cif_path', 'examples/materials/{mat1}.cif')
        preview = document.expansion_output_dirs()
        assert preview.dirs == []
        assert '[2D-2]' in preview.error
        assert '{mat2}' in preview.error


class TestMaterialEntries:
    """Pair-aware reading and writing of the material list."""

    def test_pair_entries_round_trip(self, tmp_path):
        document = ConfigDocument.blank('sheetonsheet')
        document.source_path = tmp_path / 'c.ini'
        document.write_material_names([('h_MoS2', 'h_WS2'), ('h_WS2', 'h_MoS2')])
        assert document.material_entries() == [('h_MoS2', 'h_WS2'),
                                               ('h_WS2', 'h_MoS2')]

    def test_single_names_still_write_one_per_line(self, tmp_path):
        document = ConfigDocument.blank('sheetonsheet')
        document.source_path = tmp_path / 'c.ini'
        target = document.write_material_names(['h_MoS2', 'h_WS2'])
        assert target.read_text(encoding='utf-8') == 'h_MoS2\nh_WS2\n'
        assert document.material_entries() == [('h_MoS2',), ('h_WS2',)]

    def test_a_pair_list_is_read_as_entries_not_names(self, monkeypatch):
        """read_material_list refuses a '+' line, so the single-material
        accessor must report nothing rather than raise."""
        monkeypatch.chdir(REPO_ROOT)
        document = ConfigDocument.from_file(
            EXAMPLES / 'hetero_pair_sweep.ini', 'sheetonsheet')
        assert document.material_names() == []
        assert document.material_entries() == [
            ('h-MoS2', 'h-WS2'), ('h-MoSe2', 'h-WSe2'), ('h-MoTe2', 'h-WTe2')]

    def test_an_unreadable_list_is_not_an_error(self):
        document = ConfigDocument.blank('sheetonsheet')
        document.set('2D', 'materials_list', 'nowhere.txt')
        assert document.material_entries() == []

    def test_an_entry_that_is_not_a_pair_is_refused(self, tmp_path):
        document = ConfigDocument.blank('sheetonsheet')
        document.source_path = tmp_path / 'c.ini'
        with pytest.raises(DocumentError, match='ordered pair'):
            document.write_material_names([('a', 'b', 'c')])


class TestPairBuiltInTheGui:
    """A pair assembled in the GUI -- blank config, "add a material", tick the
    sweep -- must sweep BOTH sheets. Templating only the first (which a plain
    `{mat}` does) leaves the second pinned to whatever it holds, and every run
    in the sweep then builds the same second material: wrong results from a
    config that reads as correct.
    """

    @staticmethod
    def _pair_sweep(tmp_path):
        materials = tmp_path / 'materials'
        potentials = tmp_path / 'potentials'
        for name in ('h-MoS2', 'h-WS2', 'h-MoSe2', 'h-WSe2'):
            materials.mkdir(exist_ok=True)
            potentials.mkdir(exist_ok=True)
            (materials / f'{name}.cif').write_text('data_x\n', encoding='utf-8')
            (potentials / f'{name}.sw').write_text('# sw\n', encoding='utf-8')

        document = ConfigDocument.blank('sheetonsheet')
        document.source_path = tmp_path / 'pair.ini'
        document.add_second_material()
        document.apply_material_directories(materials, potentials, '.sw')
        document.write_material_names([('h-MoS2', 'h-WS2'), ('h-MoSe2', 'h-WSe2')])
        return document

    def test_each_sheet_gets_its_own_placeholder(self, tmp_path):
        document = self._pair_sweep(tmp_path)
        assert document.get('2D', 'mat') == '{mat1}'
        assert document.get('2D-2', 'mat') == '{mat2}'
        assert document.get('2D-2', 'cif_path').endswith('{mat2}.cif')

    def test_the_second_material_actually_varies_across_the_sweep(self, tmp_path):
        from src.run.expand import expand_config_sweeps

        document = self._pair_sweep(tmp_path)
        expanded = expand_config_sweeps(document.normalised_values(),
                                        document.base_dir)

        assert [run['2D-1']['mat'] for run in expanded] == ['h-MoS2', 'h-MoSe2']
        assert [run['2D-2']['mat'] for run in expanded] == ['h-WS2', 'h-WSe2']

    @staticmethod
    def _pair_sweep_reversed(tmp_path):
        """The other click order: tick the sweep first, add the material after.
        The document must come out identical -- a config saved half-templated
        and run from the CLI never reaches the commit path that would heal it.
        """
        materials = tmp_path / 'materials'
        potentials = tmp_path / 'potentials'
        materials.mkdir(exist_ok=True)
        potentials.mkdir(exist_ok=True)
        for name in ('h-MoS2', 'h-WS2', 'h-MoSe2', 'h-WSe2'):
            (materials / f'{name}.cif').write_text('data_x\n', encoding='utf-8')
            (potentials / f'{name}.sw').write_text('# sw\n', encoding='utf-8')

        document = ConfigDocument.blank('sheetonsheet')
        document.source_path = tmp_path / 'pair.ini'
        document.apply_material_directories(materials, potentials, '.sw')
        document.add_second_material()
        document.write_material_names([('h-MoS2', 'h-WS2'), ('h-MoSe2', 'h-WSe2')])
        return document

    def test_adding_the_material_last_still_templates_both_sheets(self, tmp_path):
        document = self._pair_sweep_reversed(tmp_path)
        assert document.get('2D', 'mat') == '{mat1}'
        assert document.get('2D-2', 'mat') == '{mat2}'
        assert document.get('2D-2', 'cif_path').endswith('{mat2}.cif')
        assert document.get('2D-2', 'pot_path').endswith('{mat2}.sw')

    def test_either_click_order_gives_the_same_document(self, tmp_path):
        (tmp_path / 'a').mkdir()
        (tmp_path / 'b').mkdir()
        forward = self._pair_sweep(tmp_path / 'a')
        reversed_ = self._pair_sweep_reversed(tmp_path / 'b')
        for section in ('2D', '2D-2'):
            assert ({k: v for k, v in forward.values[section].items()
                     if k in ConfigDocument.TEMPLATED_KEYS} ==
                    {k: v for k, v in reversed_.values[section].items()
                     if k in ConfigDocument.TEMPLATED_KEYS})

    def test_the_reversed_order_sweeps_the_second_material_too(self, tmp_path):
        from src.run.expand import expand_config_sweeps

        document = self._pair_sweep_reversed(tmp_path)
        expanded = expand_config_sweeps(document.normalised_values(),
                                        document.base_dir)

        assert [run['2D-1']['mat'] for run in expanded] == ['h-MoS2', 'h-MoSe2']
        assert [run['2D-2']['mat'] for run in expanded] == ['h-WS2', 'h-WSe2']

    def test_adding_a_material_to_an_untemplated_config_stays_untemplated(
            self, tmp_path):
        """Seeding templates is only right when the first sheet already has
        them; an ordinary two-material config must not sprout placeholders."""
        document = ConfigDocument.blank('sheetonsheet')
        document.source_path = tmp_path / 'plain.ini'
        document.set('2D', 'mat', 'h-MoS2')
        document.add_second_material()
        assert document.get('2D', 'mat') == 'h-MoS2'
        assert not document.get('2D-2', 'cif_path')

    def test_a_single_material_keeps_the_original_spelling(self, tmp_path):
        """`{mat}` is what every shipped config uses; a lone sheet must not
        start writing `{mat1}` just because the pair path exists."""
        document = ConfigDocument.blank('sheetonsheet')
        document.source_path = tmp_path / 'single.ini'
        document.apply_material_directories(tmp_path / 'cif', tmp_path / 'pot', '.sw')
        assert document.get('2D', 'mat') == '{mat}'


class TestRemovingTheSecondMaterial:
    """Dropping the second sheet has to leave a config that still sweeps.

    A pair list left pointed at a one-sheet config is the silent failure:
    `read_material_list` refuses an 'a + b' line, `material_names()` swallows
    that and reports [], and the sweep quietly stops sweeping -- one run
    instead of N, with nothing reported anywhere.
    """

    @staticmethod
    def _pair_document(tmp_path, entries):
        document = ConfigDocument.blank('sheetonsheet')
        document.source_path = tmp_path / 'pair.ini'
        document.add_second_material()
        document.apply_material_directories(
            tmp_path / 'cif', tmp_path / 'pot', '.sw')
        document.write_material_names(entries)
        return document

    def test_a_pair_list_collapses_to_its_first_members(self, tmp_path):
        document = self._pair_document(
            tmp_path, [('h-MoS2', 'h-WS2'), ('t-MoS2', 'h-WS2'),
                       ('h-MoS2', 'h-WSe2')])
        # Sanity: while the list names pairs, the single-material reader
        # reports nothing at all -- which is exactly what makes the leftover
        # state invisible.
        assert document.material_names() == []

        document.remove_second_material()

        # Deduped, in file order: the first members are what the remaining
        # sheet builds, and 'h-MoS2' heads two of the three pairs. A list
        # naming it twice would build the same run twice, into the same
        # directory.
        assert document.material_names() == ['h-MoS2', 't-MoS2']
        assert (tmp_path / 'pair_materials.txt').read_text(
            encoding='utf-8') == 'h-MoS2\nt-MoS2\n'

    def test_the_configured_pair_file_is_left_untouched(self, tmp_path):
        """FIX ROUND 1: the collapse writes a NEW list beside the config
        rather than rewriting whatever the picker was pointed at. A curated
        list can live outside the config's directory and be shared by several
        configs -- collapsing it in place discards every second member for all
        of them, immediately and permanently, for an in-memory edit the user
        can still throw away by closing without saving."""
        curated = tmp_path / 'shared' / 'curated_pairs.txt'
        curated.parent.mkdir()
        original = 'h-MoS2 + h-WS2\nt-MoS2 + h-WS2\n'
        curated.write_text(original, encoding='utf-8')

        document = ConfigDocument.blank('sheetonsheet')
        document.source_path = tmp_path / 'pair.ini'
        document.add_second_material()
        document.set('2D', 'materials_list', str(curated))

        document.remove_second_material()

        assert curated.read_text(encoding='utf-8') == original
        assert document.material_names() == ['h-MoS2', 't-MoS2']
        assert document.get('2D', 'materials_list') == 'pair_materials.txt'

    def test_a_collapse_plan_names_the_two_files_and_the_loss(self, tmp_path):
        """What the confirmation prompt is built from. None when there is
        nothing to collapse, so the prompt is never raised for a config that
        would lose nothing."""
        document = self._pair_document(
            tmp_path, [('h-MoS2', 'h-WS2'), ('t-MoS2', 'h-WSe2')])

        plan = document.pair_list_collapse_plan()
        assert plan is not None
        assert plan.pairs == 2
        assert plan.dropped == ['h-WS2', 'h-WSe2']
        assert plan.target.name == 'pair_materials.txt'
        assert plan.source.name == 'pair_materials.txt'

        document.remove_second_material()
        assert document.pair_list_collapse_plan() is None

    def test_a_single_material_list_needs_no_plan(self, tmp_path):
        """Nothing is lost collapsing a list that names no pairs, so there is
        nothing to confirm -- a prompt raised here is one nobody reads."""
        document = ConfigDocument.blank('sheetonsheet')
        document.source_path = tmp_path / 'single.ini'
        document.add_second_material()
        document.write_material_names(['h-MoS2', 't-MoS2'])

        assert document.pair_list_collapse_plan() is None

    def test_a_failed_collapse_changes_nothing(self, tmp_path):
        """FIX ROUND 1: the write is the one irreversible step, and it can
        fail. Letting the exception out mid-way left [2D-2] already deleted,
        the pair list still configured and the caller's own bookkeeping never
        reached -- the exact broken state this method exists to prevent."""
        curated = tmp_path / 'shared' / 'curated_pairs.txt'
        curated.parent.mkdir()
        curated.write_text('h-MoS2 + h-WS2\n', encoding='utf-8')
        document = ConfigDocument.blank('sheetonsheet')
        document.source_path = tmp_path / 'pair.ini'
        document.add_second_material()
        document.apply_material_directories(
            tmp_path / 'cif', tmp_path / 'pot', '.sw')
        document.set('2D', 'materials_list', str(curated))
        before = {section: dict(values)
                  for section, values in document.values.items()}
        # A directory where the new list has to go: write_text raises
        # IsADirectoryError, an OSError like any other unwritable target.
        (tmp_path / 'pair_materials.txt').mkdir()

        with pytest.raises(DocumentError, match='material list'):
            document.remove_second_material()

        assert document.has_second_material()
        assert document.values == before
        assert document.get('2D', 'mat') == '{mat1}'

    def test_a_pair_list_with_nowhere_to_go_refuses_rather_than_dropping_it(
            self, tmp_path):
        """An unsaved config with no workspace has nowhere to write the
        collapsed list. Removing the sheet anyway would leave the pair list
        behind -- the silent stop-sweeping this whole fix is about.

        The list is a copy in `tmp_path`, never a file in `examples/`: run
        against the pre-fix code, this test collapsed the repository's own
        `hetero_pairs.txt` in place -- which is precisely the shared-curated-
        file hazard the new-file target now removes.
        """
        curated = tmp_path / 'pairs.txt'
        curated.write_text('h-MoS2 + h-WS2\n', encoding='utf-8')
        document = ConfigDocument.blank('sheetonsheet')
        document.set('2D', 'materials_list', str(curated))
        document.add_second_material()

        with pytest.raises(DocumentError, match='[Ss]ave the config first'):
            document.remove_second_material()
        assert document.has_second_material()

    def test_a_failed_write_leaves_the_previous_list_intact(self, tmp_path,
                                                            monkeypatch):
        """FIX ROUND 2: `write_text` truncates before it writes, so a failure
        part-way through (a full disk) left a half-written list behind while
        the caller rolled the document back to the old one -- two sources of
        truth disagreeing, and the file the CLI actually reads being the wrong
        one. The write goes to a sibling and is renamed over the target, so
        the old list survives every failure short of the rename itself.

        The failure is injected *during* the write, which is the only place it
        discriminates: a write that fails before touching the file at all
        (an unwritable target) leaves the old contents alone either way."""
        document = ConfigDocument.blank('sheetonsheet')
        document.source_path = tmp_path / 'c.ini'
        document.write_material_names(['h-MoS2', 't-MoS2'])
        listing = tmp_path / 'c_materials.txt'
        original = listing.read_text(encoding='utf-8')

        real_write_text = Path.write_text

        def _dies_midway(self, text, *args, **kwargs):
            real_write_text(self, text[:3], *args, **kwargs)
            raise OSError(28, 'No space left on device')

        monkeypatch.setattr(Path, 'write_text', _dies_midway)

        with pytest.raises(OSError):
            document.write_material_names(['b-AlSb', 'h-WS2'])

        assert listing.read_text(encoding='utf-8') == original
        assert not list(tmp_path.glob('.*.tmp'))

    def test_a_symlinked_list_is_still_followed(self, tmp_path):
        """The user pointed the config at a link on purpose; replacing the
        link with a regular file is not this code's decision to make."""
        real = tmp_path / 'real_materials.txt'
        real.write_text('h-MoS2\n', encoding='utf-8')
        link = tmp_path / 'c_materials.txt'
        link.symlink_to(real)

        document = ConfigDocument.blank('sheetonsheet')
        document.source_path = tmp_path / 'c.ini'
        document.set('2D', 'materials_list', 'c_materials.txt')
        document.write_material_names(['t-MoS2'])

        assert link.is_symlink()
        assert real.read_text(encoding='utf-8') == 't-MoS2\n'

    def test_a_single_material_list_is_left_alone(self, tmp_path):
        document = ConfigDocument.blank('sheetonsheet')
        document.source_path = tmp_path / 'single.ini'
        document.add_second_material()
        document.write_material_names(['h-MoS2', 't-MoS2'])

        document.remove_second_material()

        assert document.material_names() == ['h-MoS2', 't-MoS2']

    def test_the_primary_sheet_goes_back_to_the_plain_spelling(self, tmp_path):
        """`{mat1}` substitutes identically to `{mat}`, but a one-sheet config
        spelled for a pair reads as though the pair were still there."""
        document = self._pair_document(tmp_path, [('h-MoS2', 'h-WS2')])
        assert document.get('2D', 'mat') == '{mat1}'

        document.remove_second_material()

        assert document.get('2D', 'mat') == '{mat}'
        assert document.get('2D', 'cif_path') == 'cif/{mat}.cif'
        assert document.get('2D', 'pot_path') == 'pot/{mat}.sw'

    def test_an_untemplated_config_gains_no_placeholders(self, tmp_path):
        document = ConfigDocument.blank('sheetonsheet')
        document.source_path = tmp_path / 'plain.ini'
        document.set('2D', 'mat', 'h-MoS2')
        document.add_second_material()

        document.remove_second_material()

        assert document.get('2D', 'mat') == 'h-MoS2'


class TestApplyPairModeArgument:
    """`pair_mode` is what the caller believes; the document is what is true."""

    def test_stating_the_pair_mode_writes_the_pair_markers(self, tmp_path):
        document = ConfigDocument.blank('sheetonsheet')
        document.source_path = tmp_path / 'pair.ini'
        document.add_second_material()
        document.apply_material_directories(
            tmp_path / 'cif', tmp_path / 'pot', '.sw', pair_mode=True)
        assert document.get('2D', 'mat') == '{mat1}'
        assert document.get('2D-2', 'mat') == '{mat2}'

    def test_a_pair_mode_the_config_contradicts_is_refused(self, tmp_path):
        """Obeying `pair_mode=True` without a `[2D-2]` would conjure the
        section out of a template write; obeying `pair_mode=False` with one
        would pin the second sheet for every run of the sweep."""
        document = ConfigDocument.blank('sheetonsheet')
        document.source_path = tmp_path / 'single.ini'
        with pytest.raises(DocumentError, match='pair_mode'):
            document.apply_material_directories(
                tmp_path / 'cif', tmp_path / 'pot', '.sw', pair_mode=True)
        assert document.get('2D-2', 'mat') is None

        document.add_second_material()
        with pytest.raises(DocumentError, match='pair_mode'):
            document.apply_material_directories(
                tmp_path / 'cif', tmp_path / 'pot', '.sw', pair_mode=False)


class TestTemplateMarkers:
    """A pair config templates on {mat1}/{mat2}; a literal '{mat}' test leaves
    its placeholders behind on untick and reads back nothing on open."""

    def test_pair_templates_are_read_back(self, tmp_path):
        document = ConfigDocument.from_file(
            EXAMPLES / 'hetero_pair_sweep.ini', 'sheetonsheet')
        structures, potentials, extension = document.material_directories()
        assert structures == Path('examples/materials')
        assert potentials == Path('examples/potentials/sw/sw_lammps')
        assert extension == '.sw'

    def test_clearing_removes_pair_placeholders(self):
        document = ConfigDocument.from_file(
            EXAMPLES / 'hetero_pair_sweep.ini', 'sheetonsheet')
        assert document.clear_material_directories()
        assert document.get('2D', 'cif_path') is None
        assert document.get('2D', 'mat') is None
        assert document.get('2D-2', 'cif_path') is None
        assert document.get('2D-2', 'mat') is None


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


class TestAngleSplitWithAForceGate:
    """The panel must refuse exactly what run_simulations refuses -- and that
    depends on which builder the config routes to, which the pydantic models
    cannot see. Reported for a builder that really splits the run phase per
    angle; silent for one that keeps the in-script loop and honours the gate.
    """

    @staticmethod
    def _document(name, model):
        document = ConfigDocument.from_file(EXAMPLES / name, model)
        document.set('general', 'scan_angle', [0, 30])
        document.set('general', 'scan_angle_force', 30)
        document.set('general', 'outer_loop', 'scan_angle')
        return document

    def test_reported_where_the_split_is_real(self, monkeypatch):
        monkeypatch.chdir(REPO_ROOT)
        result = self._document('afm_flake_config.ini', 'afm').validate()

        assert not result.ok
        issues = result.issues_for('general', 'outer_loop')
        assert issues and 'scan_angle_force' in issues[0].message

    def test_silent_where_the_builder_never_splits(self, monkeypatch):
        """IndentSimulation keeps a single slide.in whatever outer_loop says,
        and afm/indent.lmp honours the gate in-script, so this config builds
        and runs correctly -- reporting it would refuse working work."""
        monkeypatch.chdir(REPO_ROOT)
        result = self._document('indent_config.ini', 'indent').validate()

        assert result.ok, result.render()


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

    def test_outer_loop_scan_angle_multiplies_jobs_for_afm(self, monkeypatch):
        """The GUI's displayed count follows src.run.expand.count_expansion, which
        Task 1 already made generic over which parameter is named by
        outer_loop -- this pins that the GUI side of that plumbing (the
        document -> count_expansion call) actually reflects it, now that the
        dropdown offers scan_angle for this mode."""
        monkeypatch.chdir(REPO_ROOT)
        document = ConfigDocument.from_file(EXAMPLES / 'afm_flake_config.ini', 'afm')
        baseline = document.expansion()
        assert (baseline.configs, baseline.builds, baseline.jobs) == (1, 1, 1)

        document.set('general', 'outer_loop', 'scan_angle')
        document.set('general', 'scan_angle', [0, 30, 60])
        split = document.expansion()
        assert (split.configs, split.builds, split.jobs) == (1, 1, 3)


class TestExpansionOutputDirs:
    """The Generate confirmation previews these dirs and flags collisions
    among them -- the exact hazard the loopability rule exists to prevent."""

    def test_one_dir_per_config_matching_the_real_layout(self, monkeypatch):
        monkeypatch.chdir(REPO_ROOT)
        document = ConfigDocument.from_file(EXAMPLES / 'indent_config.ini', 'indent')
        preview = document.expansion_output_dirs()
        assert preview.error is None
        assert len(preview.dirs) == document.expansion().configs == 1
        assert str(preview.dirs[0]) == 'indent/h-MoS2/60x_60y/sub_aSi_tip_Si_r25/K300/pot_sw_MoS2_wen'

    def test_a_sweep_gives_one_distinct_dir_per_member(self, monkeypatch):
        monkeypatch.chdir(REPO_ROOT)
        document = ConfigDocument.from_file(EXAMPLES / 'indent_config.ini', 'indent')
        document.set('general', 'temp', [300, 500])
        dirs = document.expansion_output_dirs().dirs
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
        dirs = document.expansion_output_dirs().dirs
        assert len(dirs) == 2
        assert len(set(dirs)) == 1

    def test_a_config_that_cannot_expand_reports_why(self, monkeypatch):
        """expand_config_sweeps raises when a materials_list resolves to
        materials none of which are usable (see its docstring); that is a
        config mid-edit, not a crash, so this must not propagate -- but the
        reason must come back with the empty list, not instead of it."""
        monkeypatch.chdir(REPO_ROOT)
        document = ConfigDocument.from_file(EXAMPLES / 'indent_config.ini', 'indent')
        monkeypatch.setattr(
            'src.gui.controllers.document.expand_config_sweeps',
            lambda *a, **k: (_ for _ in ()).throw(ValueError('no usable materials')))
        preview = document.expansion_output_dirs()
        assert preview.dirs == []
        assert preview.error == 'no usable materials'

    def test_a_real_pair_guard_message_reaches_the_preview(self, monkeypatch):
        """End to end, with no raise injected: a real config, broken the way a
        user actually breaks it, must arrive as the guard's own sentence.

        The other two error tests monkeypatch `expand_config_sweeps` to throw,
        which pins the handler but not the chain. This pins the chain: the
        guard lives in `core.run`, several layers below, and the failure it
        catches -- `[2D-2]` whose `mat` was updated to `{mat2}` while its
        `cif_path` still says `{mat1}` -- builds a *homostructure* under a
        correctly-named heterostructure directory. It is silent by nature, so
        the sentence explaining it is the only thing standing between the user
        and a directory of runs that are not what they asked for.
        """
        monkeypatch.chdir(REPO_ROOT)
        document = ConfigDocument.from_file(
            EXAMPLES / 'hetero_pair_sweep.ini', 'sheetonsheet')
        assert document.expansion_output_dirs().dirs, "example must expand first"

        document.set('2D-2', 'cif_path',
                     document.get('2D-2', 'cif_path').replace('{mat2}', '{mat1}'))
        preview = document.expansion_output_dirs()

        assert preview.dirs == []
        assert preview.error is not None
        # The guard names the offending section, the marker it wrongly carries,
        # the consequence, and the fix. An empty list carries none of that.
        assert '[2D-2]' in preview.error
        assert '{mat1}' in preview.error
        assert 'both sheets would build the same one' in preview.error
        assert '{mat2}' in preview.error

    def test_an_unexpected_failure_is_reported_without_raising(self, monkeypatch):
        """The preview runs on the GUI thread, so a bug in the expansion must
        not take the window down -- but it must not read as "this config
        generates nothing" either."""
        monkeypatch.chdir(REPO_ROOT)
        document = ConfigDocument.from_file(EXAMPLES / 'indent_config.ini', 'indent')
        monkeypatch.setattr(
            'src.gui.controllers.document.expand_config_sweeps',
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError('boom')))
        preview = document.expansion_output_dirs()
        assert preview.dirs == []
        assert 'boom' in preview.error


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


class TestMaterialListTarget:
    """write_material_names' fallback when there is no saved config to write
    beside. It used to fall back to Path.cwd(), dropping a materials.txt
    wherever the process happened to be launched from -- see the method's
    docstring for why that is worse than mere litter: a stray materials.txt
    in the repo root silently shadows any other config's relatively
    referenced list (src.core.utils.search_paths tries the name as given,
    i.e. against the cwd, before base_dir -- deliberately, so this ordering
    must not change)."""

    def test_declines_without_a_source_path_or_workspace_dir(self, monkeypatch,
                                                              tmp_path):
        monkeypatch.chdir(tmp_path)
        document = ConfigDocument.blank('indent')
        assert document.write_material_names(['h-MoS2']) is None
        assert document.get('2D', 'materials_list') is None
        assert not (tmp_path / 'materials.txt').exists()

    def test_writes_into_workspace_dir_when_unsaved(self, monkeypatch, tmp_path):
        monkeypatch.chdir(REPO_ROOT)
        workspace_dir = tmp_path / 'configs'
        workspace_dir.mkdir()
        document = ConfigDocument.blank('indent')
        document.workspace_dir = workspace_dir

        target = document.write_material_names(['h-MoS2', 't-MoS2'])

        assert target == workspace_dir / 'materials.txt'
        assert target.read_text(encoding='utf-8') == 'h-MoS2\nt-MoS2\n'
        assert document.get('2D', 'materials_list') == str(target)
        assert not (REPO_ROOT / 'materials.txt').exists()

    def test_a_saved_config_still_prefers_its_own_directory(self, monkeypatch,
                                                             tmp_path):
        """source_path wins over workspace_dir when both are set: the pair
        travels together beside the config, not scattered into the
        workspace's own configs directory."""
        monkeypatch.chdir(REPO_ROOT)
        document = ConfigDocument.blank('indent')
        document.source_path = tmp_path / 'run.ini'
        document.workspace_dir = tmp_path / 'somewhere_else'

        target = document.write_material_names(['h-MoS2'])

        assert target == tmp_path / 'run_materials.txt'
