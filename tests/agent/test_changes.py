"""Change sets: previewed on copies, replayed on the live documents on Apply."""
import copy
from pathlib import Path

import pytest

from src.agent.changes import (ChangeSet, Op, ToolInputError, apply_changeset,
                               apply_fs_changeset, build_changeset,
                               build_fs_changeset, describe_change, fingerprint)
from tests.helpers.agent import EXAMPLES, make_deps

SW = EXAMPLES / 'potentials' / 'sw' / 'sw_lammps'
CIF = EXAMPLES / 'materials'


@pytest.fixture(name='deps')
def _deps(tmp_path, mock_settings):
    return make_deps(tmp_path, mock_settings)


def _live(deps):
    """Stand-ins for MainWindow's live documents (deps holds snapshots)."""
    return copy.deepcopy(deps.document), copy.deepcopy(deps.settings_document)


def test_build_previews_without_touching_anything(deps):
    before = copy.deepcopy(deps.document.values)
    cs = build_changeset(deps, 'loads', [Op('set', 'general', 'force', [5, 10])])
    assert deps.document.values == before
    assert any(d.section == 'general' and d.key == 'force' and d.after == [5, 10] for d in cs.diff)
    assert list((deps.roots.workspace / 'configs').iterdir()) == []


def test_material_sweep_preview_names_the_real_target(deps):
    ops = [Op('material_dirs', value=(str(CIF), str(SW), '.sw')),
           Op('materials', value=['h-MoS2', 'h-WS2']),
           Op('set', '2D', 'pot_type', 'sw')]
    cs = build_changeset(deps, 'sweep', ops)
    assert any('workspace:/configs/materials.txt' in s and '2 names' in s for s in cs.side_effects)
    assert not (deps.roots.workspace / 'configs' / 'materials.txt').exists()
    assert not any(p.is_dir() and any(p.iterdir()) for p in (deps.roots.workspace / '.preview' / '_agent').glob('*'))
    assert not any('materials_list' in issue for issue in cs.issues)
    assert cs.expansion is not None and cs.expansion.configs >= 2


def test_apply_replays_on_live_documents_and_writes_list(deps):
    live_doc, live_settings = _live(deps)
    ops = [Op('material_dirs', value=(str(CIF), str(SW), '.sw')),
           Op('materials', value=['h-MoS2', 'h-WS2'])]
    cs = build_changeset(deps, 'sweep', ops)
    result = apply_changeset(cs, live_doc, live_settings)
    assert result.applied and not result.stale
    written = deps.roots.workspace / 'configs' / 'materials.txt'
    assert written.read_text().split() == ['h-MoS2', 'h-WS2']
    assert live_doc.material_names() == ['h-MoS2', 'h-WS2']
    assert live_doc.dirty


def test_stale_apply_refuses(deps):
    live_doc, live_settings = _live(deps)
    cs = build_changeset(deps, 'loads', [Op('set', 'general', 'force', [5])])
    live_doc.set('general', 'temp', 123)
    result = apply_changeset(cs, live_doc, live_settings)
    assert not result.applied and result.stale
    assert live_doc.get('general', 'force') != [5]
    assert apply_changeset(cs, live_doc, live_settings, force=True).applied
    assert live_doc.get('general', 'force') == [5]


def test_materials_without_target_blocks(deps):
    deps.document.workspace_dir = None
    cs = build_changeset(deps, 'sweep', [Op('materials', value=['h-MoS2'])])
    assert not cs.ok
    assert any('save the config first' in s for s in cs.issues)


@pytest.mark.parametrize('value', ['/etc/passwd', '../../outside.cif'])
def test_outside_path_refused(deps, value):
    with pytest.raises(ToolInputError, match='outside'):
        build_changeset(deps, 'bad', [Op('set', '2D', 'cif_path', value)])


def test_unknown_model_becomes_tool_input_error(deps):
    with pytest.raises(ToolInputError):
        build_changeset(deps, 'bad', [Op('set_model', value='bogus')])


def test_settings_change_previewed_and_applied(deps):
    live_doc, live_settings = _live(deps)
    section, key, old = next(
        (sec, k, v) for sec, vals in deps.settings_document.values.items()
        if isinstance(vals, dict) for k, v in vals.items()
        if isinstance(v, (int, float)))
    new = (not old) if isinstance(old, bool) else old + 1
    cs = build_changeset(deps, 's', [Op('settings_set', section, key, new)])
    assert cs.touches_settings()
    assert any(d.scope == 'settings' and d.key == key and d.after == new for d in cs.diff)
    assert apply_changeset(cs, live_doc, live_settings).applied
    assert live_settings.get(section, key) == new


def test_preview_diff_shows_real_materials_list(deps):
    ops = [Op('material_dirs', value=(str(CIF), str(SW), '.sw')),
           Op('materials', value=['h-MoS2'])]
    cs = build_changeset(deps, 'sweep', ops)
    assert not any('.preview' in str(d.after) for d in cs.diff)
    assert [d.after for d in cs.diff if d.key == 'materials_list'] == ['configs/materials.txt']


def test_failing_op_applies_nothing(deps):
    live_doc, live_settings = _live(deps)
    cs = build_changeset(deps, 'x', [Op('set', 'general', 'temp', 7)])
    cs.ops.append(Op('set_model', value='bogus'))
    before = live_doc.get('general', 'temp')
    result = apply_changeset(cs, live_doc, live_settings, force=True)
    assert not result.applied and live_doc.get('general', 'temp') == before


def test_write_oserror_is_reported(deps, monkeypatch):
    live_doc, live_settings = _live(deps)
    cs = build_changeset(deps, 's', [Op('material_dirs', value=(str(CIF), str(SW), '.sw')),
                                     Op('materials', value=['h-MoS2'])])

    def boom(*_a, **_k):
        raise PermissionError('denied')
    monkeypatch.setattr(live_doc, 'write_material_names', boom)
    result = apply_changeset(cs, live_doc, live_settings)
    assert not result.applied and 'denied' in result.message


def test_materials_list_path_refused(deps):
    with pytest.raises(ToolInputError, match='outside'):
        build_changeset(deps, 'bad', [Op('set', '2D', 'materials_list', '/etc/passwd')])


def test_model_change_flags(deps):
    cs = build_changeset(deps, 'm', [Op('set_model', value='sheetonsheet')])
    assert cs.changes_model('afm') and not cs.changes_model('sheetonsheet')
    assert any(d.section == '(model)' for d in cs.diff)


def test_handle_is_short_and_says_not_applied(deps):
    cs = build_changeset(deps, 'loads', [Op('set', 'general', 'force', [5])])
    handle = cs.handle()
    assert cs.id in handle and 'do not say it is applied' in handle
    assert len(handle) < 600


def test_describe_lists_every_change(deps):
    cs = build_changeset(deps, 'loads', [Op('set', 'general', 'force', [5, 10])])
    text = describe_change(cs)
    assert '[general] force' in text and '[5, 10]' in text


def test_fingerprint_tracks_values(deps):
    a = fingerprint(deps.document, deps.settings_document)
    deps.document.set('general', 'temp', 1)
    assert fingerprint(deps.document, deps.settings_document) != a


def test_propose_records_and_notifies(deps):
    seen = []
    deps.on_proposal = seen.append
    cs = build_changeset(deps, 'loads', [Op('set', 'general', 'force', [5])])
    handle = deps.propose(cs)
    assert deps.pending == [cs] and seen == [cs] and handle == cs.handle()


def test_fs_mkdir_and_copy(deps):
    ws = deps.roots.workspace
    change = build_fs_changeset(deps, 'copy in', [
        ('mkdir', 'materials'), ('copy', 'examples:/materials/h-MoS2.cif', 'materials/h-MoS2.cif')])
    assert change.ok, change.issues
    result = apply_fs_changeset(change, ws)
    assert result.applied
    assert (ws / 'materials' / 'h-MoS2.cif').read_text() == (CIF / 'h-MoS2.cif').read_text()


@pytest.mark.parametrize('dest', ['.preview/x.cif', '../x.cif'])
def test_fs_refuses_bad_destinations(deps, dest):
    change = build_fs_changeset(deps, 'bad', [('copy', 'examples:/materials/h-MoS2.cif', dest)])
    assert not change.ok


def test_fs_apply_rechecks_and_reports_progress(deps):
    ws = deps.roots.workspace
    change = build_fs_changeset(deps, 'two', [
        ('mkdir', 'a'), ('copy', 'examples:/materials/h-MoS2.cif', 'a/x.cif')])
    (ws / 'a').mkdir()
    (ws / 'a' / 'x.cif').write_text('mine')
    result = apply_fs_changeset(change, ws)
    assert not result.applied
    assert 'already exists' in result.message
    assert (ws / 'a' / 'x.cif').read_text() == 'mine'


def test_invalid_settings_value_is_not_applied(deps):
    live_doc, live_settings = _live(deps)
    cs = build_changeset(deps, 's', [Op('settings_set', 'simulation', 'timestep', 'abc')])
    assert not cs.ok  # the preview already says so; this is "Apply anyway"
    before = copy.deepcopy(live_settings.values)
    result = apply_changeset(cs, live_doc, live_settings)
    assert not result.applied and not result.stale
    assert result.message.startswith('Not applied: settings invalid')
    assert live_settings.values == before and not live_settings.dirty


def test_card_shows_long_values_in_full(deps):
    value = 'abc' * 100
    cs = build_changeset(deps, 'long', [Op('set', 'general', 'temp', value)])
    text = describe_change(cs)
    assert '...' not in text.split('Validation')[0]
    assert repr(value) in ''.join(text.split())
    assert max(len(line) for line in text.splitlines()) <= 120


def test_direct_handle_says_applied_and_undoable(deps):
    cs = build_changeset(deps, 'loads', [Op('set', 'general', 'force', [5])])
    handle = cs.handle(direct=True)
    assert cs.id in handle and "applied" in handle and "undo" in handle
    assert 'do not say it is applied' not in handle
    assert len(handle) < 600


def test_fs_handle_ignores_direct(deps):
    change = build_fs_changeset(deps, 'mk', [('mkdir', 'materials')])
    assert change.handle(direct=True) == change.handle()


def test_propose_passes_the_apply_mode(deps):
    cs = build_changeset(deps, 'loads', [Op('set', 'general', 'force', [5])])
    deps.direct_apply = True
    assert deps.propose(cs) == cs.handle(direct=True)


def test_applied_summary_names_fields_and_model(deps):
    from src.agent.changes import applied_summary
    ops = [Op('set_model', value='sheetonsheet')] + [
        Op('set', 'general', key, value) for key, value in
        (('force', [5]), ('temp', 250), ('scan_speed', 3), ('scan_angle', 15), ('pressure', 1))]
    cs = build_changeset(deps, 'many', ops)
    text = applied_summary(cs)
    fields = [d for d in cs.diff if d.section != '(model)']
    assert text.startswith(f'Set {len(fields)} fields: [general] ')
    assert '…' in text and text.endswith('(+ model → sheetonsheet)')
    one = build_changeset(deps, 'one', [Op('set', 'general', 'force', [5])])
    assert applied_summary(one) == 'Set 1 field: [general] force'
    model = build_changeset(deps, 'm', [Op('set_model', value='sheetonsheet')])
    assert 'model → sheetonsheet' in applied_summary(model)


def test_applied_summary_marks_settings(deps):
    from src.agent.changes import applied_summary
    cs = build_changeset(deps, 's', [Op('settings_set', 'simulation', 'timestep', 0.002)])
    assert applied_summary(cs) == 'Set 1 field: settings [simulation] timestep'


def test_direct_propose_replays_on_the_snapshot_incl_material_list(deps):
    live = _live(deps)
    deps.direct_apply = True
    sweep = build_changeset(deps, 'sweep', [
        Op('material_dirs', value=(str(CIF), str(SW), '.sw')),
        Op('materials', value=['h-MoS2', 'h-WS2']), Op('set', '2D', 'pot_type', 'sw')])
    deps.propose(sweep)
    assert deps.document.get('2D', 'pot_type') == 'sw'
    assert not (deps.roots.workspace / 'configs' / 'materials.txt').exists()  # real list untouched
    loads = build_changeset(deps, 'loads', [Op('set', 'general', 'force', [5, 10])])
    deps.propose(loads)
    assert apply_changeset(sweep, *live).applied
    result = apply_changeset(loads, *live)
    assert result.applied and not result.stale


def test_ask_mode_propose_leaves_the_snapshot_alone(deps):
    before = copy.deepcopy(deps.document.values)
    deps.propose(build_changeset(deps, 'loads', [Op('set', 'general', 'force', [5])]))
    assert deps.document.values == before


def test_direct_handle_wording(deps):
    cs = build_changeset(deps, 'loads', [Op('set', 'general', 'force', [5])])
    assert ('will be applied unless it is refused; the user sees the outcome and can '
            'undo it') in cs.handle(direct=True)


def test_partial_apply_is_reported_as_partial(deps):
    document, settings_document = _live(deps)
    configs = deps.roots.workspace / 'configs'
    ops = [Op('material_dirs', value=(str(CIF), str(SW), '.sw')),
           Op('materials', value=['h-MoS2', 'h-WS2'])]
    change = build_changeset(deps, 'sweep', ops)
    configs.chmod(0o500)
    try:
        result = apply_changeset(change, document, settings_document)
    finally:
        configs.chmod(0o700)
    assert not result.applied and result.partial
    assert 'partly changed' in result.message


def test_field_diff_is_public():
    from src.agent.changes import field_diff
    out = field_diff('config', {'general': {'a': 1, 'b': 2}}, {'general': {'a': 1, 'b': 3}})
    assert [(d.section, d.key, d.before, d.after) for d in out] == [('general', 'b', 2, 3)]


def test_display_form_path_values_become_real_paths(deps):
    shown = 'examples:/materials/h-MoS2.cif'
    cs = build_changeset(deps, 'cif', [Op('set', '2D', 'cif_path', shown)])
    value = next(op.value for op in cs.ops if op.key == 'cif_path')
    assert ':/' not in value
    base = deps.document.base_dir or deps.roots.workspace
    assert (Path(base) / value).resolve() == (CIF / 'h-MoS2.cif').resolve()


def test_display_form_outside_example_tree_refused(deps):
    with pytest.raises(ToolInputError, match='outside'):
        build_changeset(deps, 'bad', [Op('set', '2D', 'cif_path', 'examples:/../../etc/passwd')])


def test_settings_refusal_drops_the_pydantic_link(deps):
    live_doc, live_settings = _live(deps)
    cs = build_changeset(deps, 's', [Op('settings_set', 'simulation', 'timestep', 'abc')])
    assert not any('errors.pydantic.dev' in i or 'further information' in i for i in cs.issues)
    result = apply_changeset(cs, live_doc, live_settings)
    assert 'errors.pydantic.dev' not in result.message
    assert 'further information' not in result.message


def test_fs_card_shows_display_paths_never_absolute(deps):
    change = build_fs_changeset(deps, 'copy in', [
        ('mkdir', 'materials'), ('copy', 'examples:/materials/h-MoS2.cif', 'materials/h-MoS2.cif')])
    text = describe_change(change)
    assert str(deps.roots.workspace) not in text and str(EXAMPLES) not in text
    assert 'workspace:/materials' in text
    assert 'examples:/materials/h-MoS2.cif -> workspace:/materials/h-MoS2.cif' in text


def test_handles_are_capped_and_keep_their_instruction(deps):
    cs = build_changeset(deps, 'x' * 400, [Op('set', 'general', 'force', [5])])
    cs.issues = ['a very long issue ' * 40] * 3
    for direct in (False, True):
        handle = cs.handle(direct=direct)
        assert len(handle) <= 600
        assert handle.endswith('undo it.' if direct else 'do not say it is applied.')
    fs = build_fs_changeset(deps, 'y' * 400, [('copy', '/etc/passwd' * 30, 'a')] * 3)
    assert len(fs.handle()) <= 600 and fs.handle().endswith('do not say it is done.')


def test_copy_never_overwrites_a_file_that_appeared(deps, tmp_path):
    from src.agent.changes import _copy_new
    source, target = tmp_path / 'src.txt', tmp_path / 'dst.txt'
    source.write_text('new')
    target.write_text('mine')
    with pytest.raises(FileExistsError):
        _copy_new(source, target)
    assert target.read_text() == 'mine'
    fresh = tmp_path / 'fresh.txt'
    _copy_new(source, fresh)
    assert fresh.read_text() == 'new'


def test_preview_scratch_only_for_material_lists(deps):
    live_doc, live_settings = _live(deps)
    cs = build_changeset(deps, 'loads', [Op('set', 'general', 'force', [5])])
    assert apply_changeset(cs, live_doc, live_settings).applied
    assert not (deps.roots.workspace / '.preview').exists()


def test_fs_write_text_creates_a_new_file(deps):
    ws = deps.roots.workspace
    change = build_fs_changeset(deps, 'cif', [('write_text', 'materials/x.cif', 'data_x\n')])
    assert change.ok, change.issues
    assert change.shown == [('write_text', 'workspace:/materials/x.cif')]
    assert 'data_x' not in describe_change(change)
    result = apply_fs_changeset(change, ws)
    assert result.applied
    assert (ws / 'materials' / 'x.cif').read_text() == 'data_x\n'


@pytest.mark.parametrize('dest', ['.preview/x.cif', '../x.cif', 'examples:/x.cif', '/tmp/x.cif'])
def test_fs_write_text_refuses_bad_destinations(deps, dest):
    assert not build_fs_changeset(deps, 'bad', [('write_text', dest, 'x')]).ok


def test_fs_write_text_never_overwrites_at_apply(deps):
    ws = deps.roots.workspace
    change = build_fs_changeset(deps, 'cif', [('write_text', 'x.cif', 'new')])
    (ws / 'x.cif').write_text('mine')
    result = apply_fs_changeset(change, ws)
    assert not result.applied and (ws / 'x.cif').read_text() == 'mine'
