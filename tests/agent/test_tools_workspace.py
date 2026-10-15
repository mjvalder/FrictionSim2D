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


def test_describe_run_caps_the_walk(deps, monkeypatch):
    run = deps.roots.workspace / 'afm_h-MoS2' / 'l1'
    for i in range(15):
        (run / f'extra{i}.txt').write_text('x')
    text = describe_run(deps, 'afm_h-MoS2/l1', max_files=10)
    assert '10+ files' in text and 'MB' not in text
    assert 'provenance.json keys: schema, model' in text
