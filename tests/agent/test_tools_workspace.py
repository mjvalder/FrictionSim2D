"""Workspace tools: list, describe, create folders, copy found files in."""
import json

import pytest

from src.agent.changes import FsChangeSet
from src.agent.sandbox import SandboxError
from src.agent.tools.workspace import (describe_run, propose_copy_in,
                                       propose_mkdir, workspace_overview)
from tests.helpers.agent import make_deps


@pytest.fixture(name='deps')
def _deps(tmp_path, mock_settings):
    deps = make_deps(tmp_path, mock_settings)
    run = deps.roots.workspace / 'afm_h-MoS2' / 'l1'
    run.mkdir(parents=True)
    (run / 'provenance.json').write_text(json.dumps({'schema': 3, 'model': 'afm'}))
    (run / 'output_full_100x100y.json').write_text('{}')
    (deps.roots.workspace / '.preview' / 'junk').mkdir(parents=True)
    return deps


def test_overview_lists_folders_not_dot_folders(deps):
    text = workspace_overview(deps)
    assert 'afm_h-MoS2/' in text and 'configs/' in text
    assert '.preview' not in text


def test_describe_run_reports_known_files(deps):
    text = describe_run(deps, 'afm_h-MoS2/l1')
    assert 'provenance.json keys: schema, model' in text
    assert 'output_full_100x100y.json' in text


def test_describe_outside_refused(deps):
    with pytest.raises(SandboxError):
        describe_run(deps, '/etc')


def test_propose_mkdir_is_a_proposal(deps):
    handle = propose_mkdir(deps, 'materials')
    assert 'proposed' in handle and not (deps.roots.workspace / 'materials').exists()
    assert isinstance(deps.pending[-1], FsChangeSet)


def test_copy_in_plans_folder_and_copy(deps):
    propose_copy_in(deps, 'examples:/materials/h-MoS2.cif')
    ops = deps.pending[-1].ops
    assert [op[0] for op in ops] == ['mkdir', 'copy']
    assert ops[1][2].endswith('materials/h-MoS2.cif')


def test_describe_run_caps_the_walk(deps):
    run = deps.roots.workspace / 'afm_h-MoS2' / 'l1'
    for i in range(15):
        (run / f'extra{i}.txt').write_text('x')
    text = describe_run(deps, 'afm_h-MoS2/l1', max_files=10)
    assert '10+ files' in text and 'MB' not in text
    assert 'provenance.json keys: schema, model' in text


def test_mkdir_accepts_the_display_form(deps):
    propose_mkdir(deps, 'workspace:/runs/new')
    change = deps.pending[-1]
    assert change.ok, change.issues
    assert change.ops == [('mkdir', str(deps.roots.workspace / 'runs' / 'new'))]


def test_copy_in_accepts_display_form_dest(deps):
    propose_copy_in(deps, 'examples:/materials/h-MoS2.cif', dest_dir='workspace:/materials')
    change = deps.pending[-1]
    assert change.ok, change.issues
    ws = deps.roots.workspace
    assert change.ops[0] == ('mkdir', str(ws / 'materials'))
    assert change.ops[1][2] == str(ws / 'materials' / 'h-MoS2.cif')
    assert all('workspace:' not in part for op in change.ops for part in op[1:])


def test_examples_destination_refused(deps):
    propose_mkdir(deps, 'examples:/x')
    change = deps.pending[-1]
    assert not change.ok and 'examples:/' in change.issues[0]
    propose_copy_in(deps, 'examples:/materials/h-MoS2.cif', dest_dir='examples:/materials')
    assert not deps.pending[-1].ok


def test_overview_survives_an_unreadable_folder(deps):
    locked = deps.roots.workspace / 'locked'
    (locked / 'inner').mkdir(parents=True)
    locked.chmod(0)
    try:
        text = workspace_overview(deps)
    finally:
        locked.chmod(0o755)
    assert 'locked/ (unreadable)' in text and 'afm_h-MoS2/' in text


def test_overview_depth_is_clamped(deps):
    deep = deps.roots.workspace / 'd1' / 'd2' / 'd3' / 'd4' / 'd5' / 'd6'
    deep.mkdir(parents=True)
    assert 'd5/' not in workspace_overview(deps, depth=100)
    assert 'd4/' in workspace_overview(deps, depth=100)
    shallow = workspace_overview(deps, depth=0)
    assert 'd1/' in shallow and 'd2/' not in shallow


def test_overview_counts_are_capped(deps):
    from src.agent.tools import workspace as ws_tools
    big = deps.roots.workspace / 'big'
    big.mkdir()
    for i in range(ws_tools.MAX_COUNT + 5):
        (big / f'f{i}').touch()
    assert f'big/ ({ws_tools.MAX_COUNT}+ entries)' in workspace_overview(deps)


def test_overview_says_when_truncated(deps):
    for i in range(80):
        (deps.roots.workspace / f'run{i:02d}').mkdir()
    text = workspace_overview(deps)
    assert text.rstrip().endswith('… more')
