"""The portable model bundle: format, catalogue, integrity check."""
import json
import logging

from src.agent import bundles
from src.agent.bundles import (FineTune, catalogue, sha256_file, verify,
                               write_bundle)

MANIFEST = {'tag': 'qwen3.5-4b', 'base': 'unsloth/Qwen3.5-4B', 'schema_hash': 'b8b7',
            'knowledge_hash': 'ea92'}


def _bundle(tmp_path, data=b'weights'):
    gguf = tmp_path / 'w.gguf'
    gguf.write_bytes(data)
    return write_bundle(tmp_path / 'models' / 'qwen3.5-4b', gguf=gguf,
                        template='RENDERER qwen3.5\nPARSER qwen3.5', manifest=MANIFEST)


def test_write_bundle_relative_from_and_sha(tmp_path):
    dest = _bundle(tmp_path)
    text = (dest / 'Modelfile').read_text()
    assert text.splitlines()[0] == 'FROM ./model.gguf'
    assert 'RENDERER qwen3.5' in text and 'PARAMETER temperature 0' in text
    assert 'PARAMETER num_ctx 16384' in text
    manifest = json.loads((dest / 'manifest.json').read_text())
    assert manifest['sha256'] == sha256_file(dest / 'model.gguf') and manifest['size'] == 7
    assert sorted(p.name for p in dest.iterdir()) == ['Modelfile', 'manifest.json', 'model.gguf']


def test_catalogue_reads_bundles(tmp_path):
    _bundle(tmp_path)
    found = catalogue(tmp_path / 'models')
    tune = found['fs2d-assistant:qwen3.5-4b']
    assert tune.installable
    assert tune.bundle == tmp_path / 'models' / 'qwen3.5-4b' and tune.schema_hash == 'b8b7'


def test_catalogue_skips_bad_manifest(tmp_path, caplog):
    _bundle(tmp_path)
    (tmp_path / 'models' / 'broken').mkdir()
    (tmp_path / 'models' / 'broken' / 'manifest.json').write_text('{')
    (tmp_path / 'models' / 'partial').mkdir()
    partial = {k: v for k, v in MANIFEST.items() if k != 'schema_hash'}
    (tmp_path / 'models' / 'partial' / 'manifest.json').write_text(
        json.dumps({**partial, 'tag': 'partial'}))
    with caplog.at_level(logging.WARNING):
        found = catalogue(tmp_path / 'models')
    assert list(found) == ['fs2d-assistant:qwen3.5-4b']
    assert 'broken' in caplog.text and 'partial' in caplog.text


def test_catalogue_of_missing_dir_is_empty(tmp_path):
    assert catalogue(tmp_path / 'nowhere') == {}


def test_bundle_without_gguf_not_installable(tmp_path):
    dest = _bundle(tmp_path)
    (dest / 'model.gguf').unlink()
    assert not catalogue(tmp_path / 'models')['fs2d-assistant:qwen3.5-4b'].installable


def test_verify_detects_partial_gguf(tmp_path):
    dest = _bundle(tmp_path)
    tune = catalogue(tmp_path / 'models')['fs2d-assistant:qwen3.5-4b']
    assert verify(tune) is None
    (dest / 'model.gguf').write_bytes(b'weig')
    assert 'does not match' in verify(tune)


def test_published_entry_installable_without_bundle(monkeypatch, tmp_path):
    tune = FineTune(tag='x', base='b', schema_hash='s', knowledge_hash='k',
                    registry='ns/fs2d-assistant:x')
    monkeypatch.setattr(bundles, 'PUBLISHED', {'x': tune})
    found = catalogue(tmp_path / 'models')
    assert found['fs2d-assistant:x'].installable and found['fs2d-assistant:x'].bundle is None


def test_local_bundle_keeps_published_registry(monkeypatch, tmp_path):
    _bundle(tmp_path)
    tune = FineTune(tag='qwen3.5-4b', base='b', schema_hash='s', knowledge_hash='k',
                    registry='ns/fs2d-assistant:qwen3.5-4b')
    monkeypatch.setattr(bundles, 'PUBLISHED', {'qwen3.5-4b': tune})
    found = catalogue(tmp_path / 'models')['fs2d-assistant:qwen3.5-4b']
    assert found.registry == 'ns/fs2d-assistant:qwen3.5-4b' and found.bundle is not None


def test_verified_schemas_count_as_a_fit(tmp_path):
    from src.agent.bundles import FineTune
    tune = FineTune(tag='t', base='b', schema_hash='old', knowledge_hash='k',
                    verified_schemas=('new',))
    assert tune.fits('old') and tune.fits('new') and not tune.fits('other')
