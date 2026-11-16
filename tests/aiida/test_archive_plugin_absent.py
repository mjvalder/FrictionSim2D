"""An archive reads without FrictionSim2D installed (Review Focus 6, critique M5).

Monkeypatching entry-point lookups in-process does not hide the plugin:
`load_entry_point` and `importlib.metadata` still resolve `friction2d.*`. So the
import runs in a subprocess started with `-I`, with the repository (and the
editable install's path entry) removed from `sys.path` and both
`importlib.metadata.entry_points` and the `importlib_metadata` backport aiida
reads filtered to drop every FrictionSim2D entry point. It creates its own
throw-away SQLite profile under `AIIDA_PATH`, runs `verdi archive import`, and
loads every `Dict`, `FolderData` and `SinglefileData`.
"""
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytest.importorskip('aiida')
pytest.importorskip('scipy')
pytestmark = pytest.mark.aiida

from aiida import orm  # noqa: E402

from src.aiida.export import export_archive  # noqa: E402
from src.aiida.ingest import import_root  # noqa: E402
from tests.aiida.test_export import _run  # noqa: E402

REPO = Path(__file__).resolve().parents[2]

_SCRIPT = textwrap.dedent('''
    import json, os, sys

    repo, archive, work = (os.path.realpath(a) for a in sys.argv[1:4])
    sys.path[:] = [p for p in sys.path
                   if not os.path.realpath(p or os.getcwd()).startswith(repo)]

    import importlib.metadata as std_md
    import importlib_metadata as md

    def _ours(ep):
        return ep.name.startswith('friction2d') or ep.value.startswith('src.')

    def _filtered(module):
        def entry_points(**params):
            every = {(ep.group, ep.name, ep.value): ep for dist in module.distributions()
                     for ep in dist.entry_points}
            kept = module.EntryPoints([ep for ep in every.values() if not _ours(ep)])
            return kept.select(**params) if params else kept
        return entry_points

    std_md.entry_points = _filtered(std_md)
    md.entry_points = _filtered(md)

    try:
        # (`import src` alone can succeed: another package ships a bare
        # `src` namespace directory into site-packages.)
        import src.aiida.calcjob  # noqa: F401
        plugin_importable = True
    except ImportError:
        plugin_importable = False

    from aiida import orm
    from aiida.cmdline.commands.cmd_verdi import verdi
    from aiida.common.exceptions import MissingEntryPointError
    from aiida.manage.configuration import create_profile, get_config, load_profile
    from aiida.plugins import CalculationFactory

    config = get_config(create=True)
    create_profile(config, storage_backend='core.sqlite_dos',
                   storage_config={'filepath': os.path.join(work, 'storage')},
                   name='absent', email='reader@example.org')
    config.store()
    try:
        CalculationFactory('friction2d.lammps')
        entry_point_found = True
    except MissingEntryPointError:
        entry_point_found = False

    try:
        verdi(['-p', 'absent', 'archive', 'import', '--no-import-group', archive],
              standalone_mode=False)
    except SystemExit as exc:
        if exc.code:
            raise
    load_profile('absent', allow_switch=True)

    counts = {}
    for cls, read in ((orm.Dict, lambda n: n.get_dict()),
                      (orm.FolderData, lambda n: [n.base.repository.get_object_content(
                          name, mode='rb') for name in _files(n)]),
                      (orm.SinglefileData, lambda n: n.get_content(mode='rb'))):
        nodes = orm.QueryBuilder().append(cls, subclassing=False).all(flat=True)
        for node in nodes:
            read(orm.load_node(node.pk))
        counts[cls.__name__] = len(nodes)
    print(json.dumps({'plugin_importable': plugin_importable,
                      'entry_point_found': entry_point_found,
                      'calcjobs': orm.QueryBuilder().append(orm.CalcJobNode).count(),
                      'counts': counts}))
''')

_FILES = textwrap.dedent('''
    def _files(node):
        out = []
        for root, _dirs, names in node.base.repository.walk():
            out += [(root / name).as_posix() for name in names]
        return out
''')


def _counts():
    return {cls.__name__: orm.QueryBuilder().append(cls, subclassing=False).count()
            for cls in (orm.Dict, orm.FolderData, orm.SinglefileData)}


def test_archive_imports_and_loads_without_the_plugin(aiida_profile_clean, tmp_path,
                                                      monkeypatch):
    import src.aiida as aiida_pkg
    monkeypatch.setattr(aiida_pkg, 'load_aiida_profile', lambda profile=None: None)
    root = tmp_path / 'sim'
    _run(root, loads=(10.0, 30.0))
    assert import_root(root, label='camp').ok()
    archive = export_archive('camp', tmp_path / 'camp.aiida')
    expected = _counts()
    n_calcjobs = orm.QueryBuilder().append(orm.CalcJobNode).count()

    work = tmp_path / 'reader'
    work.mkdir()
    script = work / 'read_archive.py'
    script.write_text(_FILES + _SCRIPT, encoding='utf-8')
    env = {k: v for k, v in os.environ.items() if not k.startswith(('AIIDA', 'PYTHON'))}
    env['AIIDA_PATH'] = str(work)
    done = subprocess.run(
        [sys.executable, '-I', str(script), str(REPO), str(archive), str(work)],
        cwd=work, env=env, capture_output=True, text=True, timeout=300, check=False)

    assert done.returncode == 0, done.stderr[-3000:]
    result = json.loads(done.stdout.strip().splitlines()[-1])
    assert result['plugin_importable'] is False
    assert result['entry_point_found'] is False
    assert result['calcjobs'] == n_calcjobs
    assert result['counts'] == expected
    assert all(result['counts'].values())
