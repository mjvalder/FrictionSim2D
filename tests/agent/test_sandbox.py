"""Path confinement for the assistant: what it may read, and where Apply may write."""
from pathlib import Path

import pytest

from frictionsim2d.agent.sandbox import (SandboxError, SearchRoots, check_write_target,
                               display_path, find_examples_dir, resolve_readable,
                               within_roots)

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(name='roots')
def _roots(tmp_path):
    ws = tmp_path / 'ws'
    (ws / 'configs').mkdir(parents=True)
    (ws / 'configs' / 'a.ini').write_text('x')
    ex = tmp_path / 'examples'
    (ex / 'materials').mkdir(parents=True)
    (ex / 'materials' / 'h-MoS2.cif').write_text('cif')
    return SearchRoots(ws, ex)


def test_find_examples_dir_is_the_packaged_library():
    """The examples:/ tier is frictionsim2d/data, so examples:/materials/... and
    examples:/potentials/... name the files they did before the library moved."""
    found = find_examples_dir()
    assert found == REPO / 'frictionsim2d' / 'data'
    assert (found / 'materials' / 'h-MoS2.cif').is_file()
    assert (found / 'potentials' / 'sw' / 'MoS2_wen.sw').is_file()


def test_tiers_order_workspace_first(roots):
    assert [name for name, _ in roots.tiers()] == ['workspace', 'examples']


def test_relative_path_found_in_any_tier(roots):
    assert resolve_readable('configs/a.ini', roots) == roots.workspace / 'configs' / 'a.ini'
    assert resolve_readable('materials/h-MoS2.cif', roots).name == 'h-MoS2.cif'


def test_display_form_round_trips(roots):
    path = roots.examples / 'materials' / 'h-MoS2.cif'
    shown = display_path(path, roots)
    assert shown == 'examples:/materials/h-MoS2.cif'
    assert resolve_readable(shown, roots) == path


@pytest.mark.parametrize('bad', ['/etc/passwd', '../../etc/passwd', 'configs/../../x'])
def test_outside_or_escaping_paths_refused(roots, bad):
    with pytest.raises(SandboxError):
        resolve_readable(bad, roots)


def test_missing_file_refused(roots):
    with pytest.raises(SandboxError):
        resolve_readable('configs/nope.ini', roots)


def test_symlink_inside_workspace_to_scratch_is_allowed(roots, tmp_path):
    scratch = tmp_path / 'scratch'
    scratch.mkdir()
    (scratch / 'run.log').write_text('log')
    (roots.workspace / 'runs').symlink_to(scratch)
    assert resolve_readable('runs/run.log', roots).name == 'run.log'


def test_within_roots(roots):
    assert within_roots(roots.workspace / 'x' / 'y', roots)
    assert not within_roots(Path('/tmp/elsewhere'), roots)


def test_write_target_ok(roots):
    target = check_write_target('materials/new.cif', roots.workspace)
    assert target == roots.workspace / 'materials' / 'new.cif'


@pytest.mark.parametrize('bad', ['.preview/x.cif', 'a/.hidden/x', '../outside.cif', '/tmp/x.cif'])
def test_write_target_refused(roots, bad):
    with pytest.raises(SandboxError):
        check_write_target(bad, roots.workspace)


def test_write_target_never_overwrites(roots):
    with pytest.raises(SandboxError, match='already exists'):
        check_write_target('configs/a.ini', roots.workspace)


def test_display_outside(roots):
    assert display_path(Path('/etc/passwd'), roots) == '(outside the allowed folders)'


def test_write_target_refuses_symlinked_escapes(roots, tmp_path):
    ws = roots.workspace
    (ws / '.preview').mkdir()
    (ws / 'lnk').symlink_to(ws / '.preview')
    outside = tmp_path / 'outside'
    outside.mkdir()
    (ws / 'outl').symlink_to(outside)
    (ws / 'dang').symlink_to(outside / 'new.txt')
    for bad in ('lnk/x', 'outl/x', 'dang'):
        with pytest.raises(SandboxError):
            check_write_target(bad, ws)


def test_a_dot_folder_is_refused_even_when_it_links_to_an_ordinary_one(roots):
    """The lexical dot-folder check is not redundant with the realpath one: a
    `.alias` pointing at a plain folder resolves to no dot part at all."""
    ws = roots.workspace
    (ws / 'plain').mkdir()
    (ws / '.alias').symlink_to(ws / 'plain')
    with pytest.raises(SandboxError, match='dot-folders'):
        check_write_target('.alias/x.cif', ws)


def test_redact_replaces_roots_longest_first_and_home(roots, tmp_path):
    from frictionsim2d.agent.sandbox import redact
    nested = SearchRoots(roots.workspace, roots.workspace / 'ex')
    text = (f"a {roots.workspace}/configs/a.ini b {roots.workspace / 'ex'}/m.cif "
            f"c {roots.workspace} d {tmp_path}/home/me/x.txt e {roots.workspace}2/y")
    out = redact(text, nested, home=tmp_path / 'home' / 'me')
    assert 'a workspace:/configs/a.ini' in out
    assert 'b examples:/m.cif' in out
    assert 'c workspace:/ d' in out
    assert 'd ~/x.txt' in out
    assert str(tmp_path) in out.split(' e ')[1]  # a sibling of a root is not a root


def test_redact_leaves_plain_text(roots):
    from frictionsim2d.agent.sandbox import redact
    assert redact('no paths here: 5 nN', roots) == 'no paths here: 5 nN'


def test_redact_leaves_a_relative_path_that_contains_a_root(roots):
    from frictionsim2d.agent.sandbox import redact
    text = f'archive{roots.workspace}/a.ini and mirror/x{roots.workspace}/b'
    assert redact(text, roots, home='/nonexistent-home') == text
