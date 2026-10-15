"""Change sets: previewed on copies, replayed on the live documents on Apply."""
import copy

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
    section = next(iter(deps.settings_document.values))
    key = next(iter(deps.settings_document.values[section]))
    old = deps.settings_document.get(section, key)
    cs = build_changeset(deps, 's', [Op('settings_set', section, key, old)])
    assert cs.touches_settings()
    assert apply_changeset(cs, live_doc, live_settings).applied


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
