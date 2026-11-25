"""The materials pack fetch: verified downloads only, never a partial pack."""
import hashlib
import shutil
import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from frictionsim2d.core.utils.paths import get_material_path, materials_pack_dir
from frictionsim2d.data import fetch
from frictionsim2d.data.fetch import (
    MANIFEST_NAME, MANIFEST_SCHEMA, PACK_REPO, FetchError, fetch_materials, materials_available,
    read_manifest, remove_materials)

BLOCK = 'amor_Ge__0123abcd.lmp'
BLOCK_TEXT = 'LAMMPS data file\n\n3 atoms\n'
SIDECAR_TEXT = '{"schema": "amorphous-1", "material": "Ge"}\n'


@pytest.fixture(autouse=True)
def _data_home(tmp_path, monkeypatch):
    monkeypatch.setenv('XDG_DATA_HOME', str(tmp_path / 'xdg'))


def _entry(text: str) -> dict:
    data = text.encode()
    return {'sha256': hashlib.sha256(data).hexdigest(), 'size': len(data)}


def _remote(files: dict, manifest_files: dict = None) -> dict:
    """What the dataset repo serves: ``{name: text}`` plus its manifest."""
    listed = manifest_files if manifest_files is not None else {
        name: _entry(text) for name, text in files.items()}
    return {**files, MANIFEST_NAME: json.dumps({'schema': MANIFEST_SCHEMA, 'files': listed})}


def _downloader(remote: dict, calls: list, fail_on: str = None):
    def download(filename: str, folder: Path) -> Path:
        calls.append(filename)
        if filename == fail_on:
            raise ConnectionError('network went away')
        dest = folder / filename
        dest.write_text(remote[filename], encoding='utf-8')
        return dest
    return download


GOOD = {BLOCK: BLOCK_TEXT, BLOCK + '.json': SIDECAR_TEXT}


def test_pack_block_found_after_fetch():
    calls = []
    with pytest.raises(FileNotFoundError):
        get_material_path(BLOCK)
    folder = fetch_materials(lambda line: None, download=_downloader(_remote(GOOD), calls))
    assert folder == materials_pack_dir()
    assert get_material_path(BLOCK) == folder / BLOCK
    assert get_material_path(f'materials/{BLOCK}') == folder / BLOCK
    assert materials_available()
    assert calls == [MANIFEST_NAME, BLOCK, BLOCK + '.json']


def test_partial_fetch_not_used():
    """A download that dies half-way leaves no pack (and no staging) behind."""
    with pytest.raises(FetchError, match='network went away'):
        fetch_materials(lambda line: None,
                        download=_downloader(_remote(GOOD), [], fail_on=BLOCK + '.json'))
    assert not materials_pack_dir().exists()
    assert not any(materials_pack_dir().parent.glob('*.partial'))
    assert not materials_available()
    with pytest.raises(FileNotFoundError):
        get_material_path(BLOCK)


def test_a_bad_sha_is_refused_and_the_installed_pack_kept():
    fetch_materials(lambda line: None, download=_downloader(_remote(GOOD), []))
    tampered = dict(GOOD, **{BLOCK: BLOCK_TEXT + 'tampered\n'})
    listed = {name: _entry(text) for name, text in GOOD.items()}
    listed[BLOCK] = {**_entry(BLOCK_TEXT + 'tampered\n'), 'sha256': '0' * 64}
    with pytest.raises(FetchError, match='sha256 does not match'):
        fetch_materials(lambda line: None, download=_downloader(_remote(tampered, listed), []))
    assert (materials_pack_dir() / BLOCK).read_text(encoding='utf-8') == BLOCK_TEXT


def test_a_wrong_size_is_refused():
    listed = {name: _entry(text) for name, text in GOOD.items()}
    listed[BLOCK]['size'] += 1
    with pytest.raises(FetchError, match='bytes, the manifest says'):
        fetch_materials(lambda line: None, download=_downloader(_remote(GOOD, listed), []))
    assert not materials_pack_dir().exists()


def test_a_verified_file_is_not_downloaded_again():
    fetch_materials(lambda line: None, download=_downloader(_remote(GOOD), []))
    calls = []
    fetch_materials(lambda line: None, download=_downloader(_remote(GOOD), calls))
    assert calls == [MANIFEST_NAME]
    assert materials_available()


def test_a_refetch_drops_files_the_new_manifest_no_longer_lists():
    fetch_materials(lambda line: None, download=_downloader(_remote(GOOD), []))
    fetch_materials(lambda line: None, download=_downloader(_remote({}), []))
    assert not (materials_pack_dir() / BLOCK).exists()
    assert materials_available()


@pytest.mark.parametrize('name', ['../escape.lmp', 'sub/dir.lmp', MANIFEST_NAME])
def test_a_manifest_naming_a_non_flat_file_is_refused(tmp_path, name):
    path = tmp_path / MANIFEST_NAME
    path.write_text(json.dumps({'schema': MANIFEST_SCHEMA,
                                'files': {name: _entry('x')}}), encoding='utf-8')
    with pytest.raises(FetchError, match='flat names'):
        read_manifest(path)


def test_a_manifest_of_another_schema_is_refused(tmp_path):
    path = tmp_path / MANIFEST_NAME
    path.write_text(json.dumps({'schema': 'other', 'files': {}}), encoding='utf-8')
    with pytest.raises(FetchError, match=MANIFEST_SCHEMA):
        read_manifest(path)


def test_available_needs_every_listed_file_at_its_size():
    fetch_materials(lambda line: None, download=_downloader(_remote(GOOD), []))
    (materials_pack_dir() / BLOCK).write_text('short', encoding='utf-8')
    assert not materials_available()


def test_remove_deletes_the_pack():
    fetch_materials(lambda line: None, download=_downloader(_remote(GOOD), []))
    assert remove_materials() is True
    assert not materials_pack_dir().exists()
    assert remove_materials() is False


def test_the_default_downloader_targets_the_dataset_repo(monkeypatch, tmp_path):
    """Without huggingface_hub, files come from the dataset's resolve/main URL."""
    import builtins
    real_import = builtins.__import__

    def no_hub(name, *args, **kwargs):
        if name == 'huggingface_hub':
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    seen = []

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self, *args):
            return b''

    monkeypatch.setattr(builtins, '__import__', no_hub)
    monkeypatch.setattr(fetch.urllib.request, 'urlopen',
                        lambda url, timeout: seen.append(url) or _Response())
    fetch.default_download()(BLOCK, tmp_path)
    assert seen == [f'https://huggingface.co/datasets/{PACK_REPO}/resolve/main/{BLOCK}']
    assert PACK_REPO == 'mjvalder/fs2d-materials'


def test_cli_fetch_materials_reports_a_failure_cleanly(monkeypatch):
    from frictionsim2d import cli as cli_module

    def boom(progress):
        raise FetchError('huggingface.co/datasets/x holds no manifest')
    monkeypatch.setattr(fetch, 'fetch_materials', boom)
    result = CliRunner().invoke(cli_module.cli, ['fetch', 'materials'])
    assert result.exit_code == 1
    assert 'holds no manifest' in result.output
    assert 'Traceback' not in result.output


def test_cli_fetch_materials_installs_and_removes(monkeypatch):
    from frictionsim2d import cli as cli_module
    monkeypatch.setattr(fetch, 'fetch_materials', lambda progress: fetch_materials(
        progress, download=_downloader(_remote(GOOD), [])))
    runner = CliRunner()
    result = runner.invoke(cli_module.cli, ['fetch', 'materials'])
    assert result.exit_code == 0, result.output
    assert f'Installed the materials pack in {materials_pack_dir()}' in result.output
    result = runner.invoke(cli_module.cli, ['fetch', 'materials', '--remove'])
    assert result.exit_code == 0 and 'Removed the materials pack' in result.output


def _make_manifest_module():
    import importlib.util
    script = Path(__file__).resolve().parents[2] / 'tools' / 'materials_pack' / 'make_manifest.py'
    spec = importlib.util.spec_from_file_location('make_manifest', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_staged_pack_round_trips_through_make_manifest_and_fetch(tmp_path):
    """tools/materials_pack/make_manifest.py writes what fetch_materials checks."""
    staged = tmp_path / 'staged'
    staged.mkdir()
    (staged / BLOCK).write_text(BLOCK_TEXT, encoding='utf-8')
    sidecar = {'schema': 'amorphous-1', 'material': 'Ge',
               'block_sha256': hashlib.sha256(BLOCK_TEXT.encode()).hexdigest()}
    (staged / f'{BLOCK}.json').write_text(json.dumps(sidecar), encoding='utf-8')
    (staged / 'README.md').write_text('card', encoding='utf-8')
    assert _make_manifest_module().main([str(staged)]) == 0
    assert set(read_manifest(staged / MANIFEST_NAME)) == {BLOCK, f'{BLOCK}.json'}

    def from_staged(filename, folder):
        shutil.copy(staged / filename, folder / filename)
        return folder / filename
    fetch_materials(lambda line: None, download=from_staged)
    assert get_material_path(BLOCK) == materials_pack_dir() / BLOCK


@pytest.mark.parametrize('case', ['no sidecar', 'wrong block sha'])
def test_make_manifest_refuses_an_inconsistent_block(tmp_path, case, capsys):
    (tmp_path / BLOCK).write_text(BLOCK_TEXT, encoding='utf-8')
    if case == 'wrong block sha':
        (tmp_path / f'{BLOCK}.json').write_text(json.dumps({'block_sha256': '0' * 64}),
                                                encoding='utf-8')
    assert _make_manifest_module().main([str(tmp_path)]) == 1
    assert not (tmp_path / MANIFEST_NAME).exists()
    assert 'error:' in capsys.readouterr().err
