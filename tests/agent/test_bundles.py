"""The published fine-tunes and the bundle format: manifest, integrity check."""
import json
import logging

from frictionsim2d.agent import bundles
from frictionsim2d.agent.bundles import (PUBLISHED, FineTune, fits_published, model_name,
                                         published_tag, read_manifest, sha256_file, verify)

MANIFEST = {'tag': 'qwen3.5-4b', 'base': 'unsloth/Qwen3.5-4B', 'schema_hash': 'b8b7',
            'knowledge_hash': 'ea92'}


def write_bundle(folder, data=b'weights', **manifest):
    """A bundle as the Hugging Face repos hold it."""
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'model.gguf').write_bytes(data)
    (folder / 'Modelfile').write_text('FROM ./model.gguf\nRENDERER qwen3.5\nPARSER qwen3.5\n'
                                      'PARAMETER temperature 0\nPARAMETER num_ctx 16384\n')
    fields = {**MANIFEST, 'sha256': sha256_file(folder / 'model.gguf'), 'size': len(data),
              **manifest}
    (folder / 'manifest.json').write_text(json.dumps(fields))
    return folder


def test_published_repos_are_the_three_qwen35_fine_tunes():
    assert PUBLISHED == {'qwen3.5-2b': 'mjvalder/fs2d-assistant-qwen3.5-2b',
                         'qwen3.5-4b': 'mjvalder/fs2d-assistant-qwen3.5-4b',
                         'qwen3.5-9b': 'mjvalder/fs2d-assistant-qwen3.5-9b'}


def test_names_and_tags():
    assert model_name('qwen3.5-4b') == 'fs2d-assistant:qwen3.5-4b'
    assert published_tag('fs2d-assistant:qwen3.5-9b') == 'qwen3.5-9b'
    assert published_tag('fs2d-assistant:q35-4b') is None
    assert published_tag('qwen3.5:4b') is None


def test_published_schemas_say_what_each_model_fits():
    assert fits_published('qwen3.5-4b', '58d9a9a68b61b3d2')
    assert fits_published('qwen3.5-4b', 'b8b78921bc6dd497')
    assert not fits_published('qwen3.5-2b', 'b8b78921bc6dd497')
    assert not fits_published('nope', '58d9a9a68b61b3d2')


def test_models_dir_is_under_xdg_data_home(monkeypatch, tmp_path):
    monkeypatch.setenv('XDG_DATA_HOME', str(tmp_path))
    assert bundles.MODELS_DIR == bundles.models_dir() == tmp_path / 'FrictionSim2D' / 'models'


def test_read_manifest(tmp_path):
    tune = read_manifest(write_bundle(tmp_path / 'qwen3.5-4b', verified_schemas=['58d9']))
    assert (tune.tag, tune.schema_hash, tune.size) == ('qwen3.5-4b', 'b8b7', 7)
    assert tune.verified_schemas == ('58d9',) and tune.bundle == tmp_path / 'qwen3.5-4b'
    assert tune.name == 'fs2d-assistant:qwen3.5-4b'


def test_bad_manifest_is_skipped_with_a_warning(tmp_path, caplog):
    (tmp_path / 'broken').mkdir()
    (tmp_path / 'broken' / 'manifest.json').write_text('{')
    (tmp_path / 'partial').mkdir()
    (tmp_path / 'partial' / 'manifest.json').write_text(json.dumps({'tag': 'x', 'base': 'b'}))
    with caplog.at_level(logging.WARNING):
        assert read_manifest(tmp_path / 'broken') is None
        assert read_manifest(tmp_path / 'partial') is None
    assert 'broken' in caplog.text and 'partial' in caplog.text


def test_verify_detects_partial_gguf(tmp_path):
    folder = write_bundle(tmp_path / 'qwen3.5-4b')
    tune = read_manifest(folder)
    assert verify(tune) is None
    (folder / 'model.gguf').write_bytes(b'weig')
    assert 'does not match' in verify(tune)
    (folder / 'model.gguf').write_bytes(b'WEIGHTS')  # same size, other bytes
    assert 'does not match' in verify(tune)


def test_verified_schemas_count_as_a_fit():
    tune = FineTune(tag='t', base='b', schema_hash='old', verified_schemas=('new',))
    assert tune.fits('old') and tune.fits('new') and not tune.fits('other')


def test_bundles_use_the_one_sha256_file():
    from frictionsim2d.data import provenance
    assert bundles.sha256_file is provenance.sha256_file
