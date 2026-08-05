"""Schema introspection: the mechanism that keeps the GUI from needing edits.

Forms are derived from the config models at run time, so a field added to a
config model appears in the GUI on its own. The coverage test is the canary: if
it reports an unclassified field, the introspection needs a new kind; if it
reports a missing override, someone should add a label. Only the first is a
failure.
"""

from __future__ import annotations

import pytest

from src.core.config import AFMSimulationConfig, SheetOnSheetSimulationConfig
from src.gui.schema.introspect import (
    EXCLUDED_ROOT_FIELDS,
    FieldKind,
    describe_model,
    describe_section,
)
from src.gui.schema.overrides import overrides_for, section_override
from src.gui.schema.simulation_types import (
    FAMILIES,
    MODES,
    family_for_model,
    mode_for_model,
    modes_for_family,
    sections_for,
)

ROOT_MODELS = [AFMSimulationConfig, SheetOnSheetSimulationConfig]
ALL_MODELS = sorted(MODES)


class TestSimulationTypes:
    """Two families with modes, not five flat types."""

    def test_two_families(self):
        assert [f.key for f in FAMILIES] == ['afm', 'sheetonsheet']

    def test_every_cli_model_is_reachable(self):
        assert set(MODES) == {'afm', 'indent', 'pes_tip', 'sheetonsheet', 'pes_sheet'}

    def test_afm_family_modes(self):
        assert [m.model for m in modes_for_family('afm')] == ['afm', 'indent', 'pes_tip']

    def test_sheet_family_modes(self):
        assert [m.model for m in modes_for_family('sheetonsheet')] == \
            ['sheetonsheet', 'pes_sheet']

    def test_only_pes_tip_lacks_a_system_phase(self):
        """It writes no system.in at all; the others share the indentation phase."""
        without = [m for _, m in MODES.values() if not m.has_system_phase]
        assert [m.model for m in without] == ['pes_tip']

    def test_only_the_afm_family_has_layer_levels(self):
        assert family_for_model('indent').has_layer_levels
        assert not family_for_model('pes_sheet').has_layer_levels

    @pytest.mark.parametrize('model', ALL_MODELS)
    def test_sections_include_the_family_basics(self, model):
        sections = sections_for(model)
        assert '2D' in sections and 'general' in sections

    def test_mode_specific_sections(self):
        assert 'indent' in sections_for('indent')
        assert 'pes' in sections_for('pes_tip')
        assert 'pes' in sections_for('pes_sheet')
        assert 'indent' not in sections_for('afm')

    def test_tip_only_in_the_afm_family(self):
        assert 'tip' in sections_for('afm')
        assert 'tip' not in sections_for('sheetonsheet')


class TestIntrospectionCoverage:
    """The canary for config-model changes."""

    @pytest.mark.parametrize('model', ROOT_MODELS, ids=lambda m: m.__name__)
    def test_no_field_is_unclassified(self, model):
        unknown = [
            f"[{section.key}] {f.key} ({f.annotation})"
            for section in describe_model(model)
            for f in section.fields
            if f.kind is FieldKind.UNKNOWN
        ]
        assert not unknown, (
            "introspection has no widget kind for these fields; add one to "
            "FieldKind and _classify:\n  " + "\n  ".join(unknown)
        )

    @pytest.mark.parametrize('model', ROOT_MODELS, ids=lambda m: m.__name__)
    def test_every_field_appears_in_exactly_one_section(self, model):
        seen = {}
        for section in describe_model(model):
            for f in section.fields:
                seen.setdefault((section.key, f.key), 0)
                seen[(section.key, f.key)] += 1
        assert all(count == 1 for count in seen.values())

    @pytest.mark.parametrize('model', ROOT_MODELS, ids=lambda m: m.__name__)
    def test_derived_fields_are_excluded(self, model):
        """settings belongs to the Settings tab; sheets is folded from [2D-N]
        sections and has no .ini section to write."""
        keys = {s.name for s in describe_model(model)}
        assert not (keys & EXCLUDED_ROOT_FIELDS)

    @pytest.mark.parametrize('model', ROOT_MODELS, ids=lambda m: m.__name__)
    def test_override_gaps_are_reported_not_fatal(self, model, capsys):
        """A field without a label still renders. This reports the backlog."""
        missing = []
        for section in describe_model(model):
            declared = section_override(section.key).fields
            for f in section.fields:
                if f.key not in declared and not f.doc:
                    missing.append(f"[{section.key}] {f.key}")
        if missing:
            print("fields with neither an override label nor a description:")
            for item in missing:
                print(f"  {item}")

    @pytest.mark.parametrize('model', ROOT_MODELS, ids=lambda m: m.__name__)
    def test_the_material_list_is_an_introspected_field(self, model):
        """It drives the material sweep, and the form is built from
        model_fields -- a key the model does not declare cannot be edited."""
        sheet = {s.key: s for s in describe_model(model)}['2D']
        assert 'materials_list' in sheet.field_map()


class TestSectionStructure:

    def test_afm_sections(self):
        keys = [s.key for s in describe_model(AFMSimulationConfig)]
        assert keys == ['general', 'tip', 'sub', '2D', 'flake', 'pes',
                        'indent', 'lattice_match', 'lj_override']

    def test_sheet_sections_have_no_tip(self):
        keys = [s.key for s in describe_model(SheetOnSheetSimulationConfig)]
        assert 'tip' not in keys

    def test_required_sections_are_not_optional(self):
        sections = {s.key: s for s in describe_model(AFMSimulationConfig)}
        assert not sections['2D'].optional
        assert not sections['tip'].optional
        assert not sections['general'].optional

    def test_optional_sections_are_marked(self):
        sections = {s.key: s for s in describe_model(AFMSimulationConfig)}
        for key in ('sub', 'flake', 'pes', 'indent', 'lattice_match'):
            assert sections[key].optional, key

    def test_section_key_is_the_ini_section_name(self):
        """[2D], not [sheet]."""
        sections = {s.name: s for s in describe_model(AFMSimulationConfig)}
        assert sections['sheet'].key == '2D'


class TestFieldClassification:

    @pytest.fixture(name='afm')
    def _afm(self):
        return {s.key: s.field_map() for s in describe_model(AFMSimulationConfig)}

    @pytest.mark.parametrize('section,key,kind', [
        ('2D', 'mat', FieldKind.STR),
        ('2D', 'pot_path', FieldKind.PATH),
        ('2D', 'cif_path', FieldKind.PATH),
        ('2D', 'layers', FieldKind.LIST),
        ('2D', 'lat_c', FieldKind.FLOAT),
        ('tip', 'r', FieldKind.FLOAT),
        ('tip', 'amorph', FieldKind.ENUM),
        ('general', 'temp', FieldKind.FLOAT),
        ('general', 'finite_sheet', FieldKind.BOOL),
        ('general', 'outer_loop', FieldKind.ENUM),
        ('general', 'twist_max_supercell', FieldKind.INT),
    ])
    def test_kind(self, afm, section, key, kind):
        assert afm[section][key].kind is kind

    def test_scalar_or_list_where_arity_changes_meaning(self, afm):
        """force can be one load or a set to loop over."""
        assert afm['general']['force'].kind is FieldKind.SCALAR_OR_LIST
        assert afm['general']['force'].item_kind is FieldKind.FLOAT

    def test_list_item_kind_is_captured(self, afm):
        assert afm['2D']['layers'].item_kind is FieldKind.INT

    def test_dict_item_kind_is_captured(self):
        """output.dump is Dict[str, bool]: KeyValueGridEditor.add_row needs
        this to give a newly added row a checkbox instead of a text box."""
        from src.core.config import GlobalSettings

        settings = {s.key: s.field_map() for s in describe_model(GlobalSettings)}
        assert settings['output']['dump'].item_kind is FieldKind.BOOL
        assert settings['output']['dump_frequency'].item_kind is FieldKind.INT

    def test_enum_choices_are_captured(self, afm):
        assert set(afm['tip']['amorph'].choices) == {'a', 'c'}
        assert set(afm['general']['outer_loop'].choices) == \
            {'pressure', 'scan_speed', 'force', 'scan_angle'}

    def test_required_flags(self, afm):
        assert afm['2D']['mat'].required
        assert not afm['general']['temp'].required

    def test_defaults_are_captured(self, afm):
        assert afm['general']['temp'].default == 300.0
        assert afm['general']['finite_sheet'].default is False

    def test_descriptions_come_through(self, afm):
        assert 'radius' in afm['tip']['r'].doc.lower()


class TestCaptionBudget:
    """Captions render permanently under every control. Long ones are why the
    form read as a wall of grey text; the long form belongs in the tooltip."""

    def _all_overrides(self):
        from src.gui.schema.overrides import SETTINGS_MODEL

        for model in list(ALL_MODELS) + [SETTINGS_MODEL]:
            for section_key, section in overrides_for(model).items():
                yield model, section_key, None, section
                for key, field in section.fields.items():
                    yield model, section_key, key, field

    def test_every_caption_is_one_short_line(self):
        from src.gui.schema.overrides import CAPTION_BUDGET

        offenders = [
            f"{model} [{section}] {key or '(section)'}: {len(entry.help)} chars"
            for model, section, key, entry in self._all_overrides()
            if entry.help and (
                len(entry.help) > CAPTION_BUDGET or '\n' in entry.help)
        ]
        assert not offenders, (
            "these captions exceed CAPTION_BUDGET or wrap; move the long form "
            "to `detail`:\n  " + "\n  ".join(offenders))


class TestRebuiltOverridesKeepTheirMetadata:
    """`overrides_for` reconstructs three entries -- the substrate section, the
    general section, and `outer_loop` when narrowing its per-family choices.
    Each copies attributes by name, so anything left out of the copy is
    dropped silently. `sub` and `general` reconstruct only the SectionOverride
    wrapper -- their `fields` dict passes through by reference, so it is each
    section's own attributes (label, help, detail, order) that are at risk,
    not anything inside `fields`. `outer_loop` is the opposite: it reconstructs
    a FieldOverride itself, so its own attributes (detail, editor, ...) are
    what a copy can drop."""

    def test_the_rebuilt_substrate_keeps_its_section_metadata(self):
        """The `sub` rebuild copies the section's own attributes by name and
        adds `required_reason`. Its `fields` dict passes through by reference,
        so the section attributes are the only thing this site can drop."""
        source = section_override('sub')
        rebuilt = overrides_for('afm')['sub']
        assert rebuilt.required_reason          # it *was* rebuilt
        assert (rebuilt.label, rebuilt.help, rebuilt.detail, rebuilt.order) == \
               (source.label, source.help, source.detail, source.order)

    def test_the_rebuilt_general_section_keeps_its_section_metadata(self):
        source = section_override('general')
        rebuilt = overrides_for('afm')['general']
        assert set(rebuilt.fields) < set(source.fields)   # it *was* rebuilt
        assert (rebuilt.label, rebuilt.help, rebuilt.detail, rebuilt.order) == \
               (source.label, source.help, source.detail, source.order)

    def test_the_narrowed_outer_loop_keeps_its_detail(self):
        """The nested rebuild: `choices` is replaced per mode (falling back to
        the family), and everything else has to be carried across by hand."""
        outer_loop = overrides_for('afm')['general'].fields['outer_loop']
        assert outer_loop.choices == ('force', 'scan_angle')   # it *was* rebuilt
        assert 'job count' in outer_loop.detail


class TestOuterLoopChoicesNarrowByMode:
    """`scan_angle` only does something in the two modes whose builder emits
    one run script per angle (dynamic_indent/'afm' and dynamic/'sheetonsheet').
    'indent' shares the AFM family but its template never reads the angle
    variable (SUPPORTS_ANGLE_SPLIT = False); both PES scan modes override
    write_inputs entirely and never look at outer_loop at all. Offering
    scan_angle there would let someone configure a split that silently does
    nothing -- exactly what outer_loop_choices exists to prevent."""

    def test_scan_angle_offered_where_the_split_is_real(self):
        assert 'scan_angle' in overrides_for('afm')['general'].fields['outer_loop'].choices
        assert 'scan_angle' in \
            overrides_for('sheetonsheet')['general'].fields['outer_loop'].choices

    def test_scan_angle_not_offered_where_the_split_is_a_no_op(self):
        assert 'scan_angle' not in overrides_for('indent')['general'].fields['outer_loop'].choices
        assert 'scan_angle' not in overrides_for('pes_tip')['general'].fields['outer_loop'].choices
        assert 'scan_angle' not in \
            overrides_for('pes_sheet')['general'].fields['outer_loop'].choices

    def test_unnarrowed_choices_are_unaffected(self):
        """`indent` still offers `force` exactly as before -- this task does
        not touch that narrowing."""
        assert overrides_for('indent')['general'].fields['outer_loop'].choices == ('force',)

    def test_pes_modes_offer_no_split_at_all(self):
        """Neither PES scan reaches the split code path (see
        src/builders/pes_scan.py -- both write_inputs overrides render a
        single ``lammps/slide.in`` directly, without ever branching on
        ``outer_loop``), so unlike `indent` (which still narrows to `force`)
        the PES modes narrow all the way down to zero choices -- not `force`,
        not `pressure`/`scan_speed`, nothing selectable."""
        assert overrides_for('pes_tip')['general'].fields['outer_loop'].choices == ()
        assert overrides_for('pes_sheet')['general'].fields['outer_loop'].choices == ()


class TestOverrides:

    def test_units_are_angstrom_not_nm(self):
        """The example configs said nm; they were wrong."""
        sheet = section_override('2D')
        assert sheet.fields['x'].unit == 'Å'
        assert section_override('sub').fields['thickness'].unit == 'Å'
        assert section_override('tip').fields['r'].unit == 'Å'

    def test_substrate_is_locked_on_except_for_pes_tip(self):
        """Every other model aborts without a substrate, so offering to remove
        it would offer a guaranteed failure."""
        for model in ('afm', 'indent', 'sheetonsheet', 'pes_sheet'):
            assert overrides_for(model)['sub'].required_reason
        assert overrides_for('pes_tip')['sub'].required_reason is None

    def test_sweep_semantics_are_explained_where_they_bite(self):
        """Long by nature, so it lives in `detail`, not the caption."""
        general = section_override('general').fields
        assert 'job count' in general['outer_loop'].detail

    def test_tip_radius_carries_the_pbc_rule(self):
        assert 'periodic image' in section_override('tip').fields['r'].detail

    def test_unknown_section_yields_an_empty_override(self):
        assert section_override('not_a_section').fields == {}

    def test_scan_speed_is_metres_per_second(self):
        from src.gui.schema.overrides import field_override
        fo = field_override('sheetonsheet', 'general', 'scan_speed')
        assert fo.unit == 'm/s'
        assert '100 m/s = 1 Å/ps' in (fo.help or '') + (fo.detail or '')

    def test_spring_constants_have_units(self):
        from src.gui.schema.overrides import field_override
        assert field_override('sheetonsheet', 'general', 'driving_spring').unit == 'N/m'
        assert field_override('sheetonsheet', 'general', 'bond_spring').unit == 'N/m'


class TestDescribeSection:

    def test_describes_an_arbitrary_model(self):
        from src.core.config import TipConfig

        spec = describe_section(TipConfig, name='tip', key='tip')
        assert spec.key == 'tip'
        assert 'r' in spec.field_map()
