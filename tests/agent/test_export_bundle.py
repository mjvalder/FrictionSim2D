"""tools/agent_data/export.py writes a portable bundle (no training stack needed)."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools' / 'agent_data'))
import export  # noqa: E402  pylint: disable=import-error,wrong-import-position
sys.path.pop(0)

METRICS = {'run': 'q35-4b', 'base': 'unsloth/Qwen3.5-4B',
           'schema_hash': 'b8b78921bc6dd497', 'knowledge_hash': 'ea92dc8b1bc8424b'}
MODELFILE = """FROM /home/mv923/FrictionSim2D-main/tools/agent_data/runs/q35-4b/gguf_gguf/X.Q4_K_M.gguf
RENDERER qwen3.5
PARSER qwen3.5
PARAMETER temperature 0
PARAMETER num_ctx 16384
# fs2d-assistant trained for tool-schema hash b8b78921bc6dd497
"""


@pytest.fixture(name='runs')
def _runs(tmp_path, monkeypatch):
    run = tmp_path / 'runs' / 'q35-4b'
    (run / 'gguf_gguf').mkdir(parents=True)
    (run / 'gguf_gguf' / 'X.Q4_K_M.gguf').write_bytes(b'weights')
    (run / 'gguf_gguf' / 'X.BF16-mmproj.gguf').write_bytes(b'vision')
    (run / 'metrics.json').write_text(json.dumps(METRICS))
    (run / 'Modelfile').write_text(MODELFILE)
    monkeypatch.setattr(export, 'RUNS', tmp_path / 'runs')
    return tmp_path


def _main(runs, *extra):
    return export.main(['--run', 'q35-4b', '--tag', 'q35-4b', '--bundle-only',
                        '--models-dir', str(runs / 'models'), *extra])


def test_bundle_only_writes_portable_bundle(runs):
    assert _main(runs) == 0
    bundle = runs / 'models' / 'q35-4b'
    text = (bundle / 'Modelfile').read_text()
    assert text.splitlines()[0] == 'FROM ./model.gguf'
    assert 'RENDERER qwen3.5' in text and 'PARSER qwen3.5' in text
    assert '/home/mv923' not in text and text.count('PARAMETER temperature') == 1
    manifest = json.loads((bundle / 'manifest.json').read_text())
    assert manifest['schema_hash'] == METRICS['schema_hash']
    assert manifest['knowledge_hash'] == METRICS['knowledge_hash']
    assert 'trained_knowledge' not in manifest and manifest['registry'] is None
    assert manifest['pass1'] is None and manifest['latency_p50'] is None
    assert (bundle / 'model.gguf').read_bytes() == b'weights'
    assert not any('mmproj' in p.name for p in bundle.iterdir())


def test_bundle_only_runs_no_subprocess(runs, monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError(f'subprocess called: {args}')
    monkeypatch.setattr(subprocess, 'run', refuse)
    assert _main(runs) == 0


def test_eval_fills_pass1_and_p50(runs):
    records = [{'case': f'c{i}', 'group': 'g', 'outcome': outcome, 'latency_s': latency,
                'cold': False, 'requests': 1, 'input_tokens': 10, 'output_tokens': 5}
               for i, (outcome, latency) in enumerate(
                   [('passed', 2), ('passed', 4), ('passed', 6), ('failed', 8)])]
    results = runs / 'bench.json'
    results.write_text(json.dumps({'provider': 'ollama', 'model': 'fs2d-assistant:q35-4b',
                                   'records': records}))
    assert _main(runs, '--eval', str(results)) == 0
    manifest = json.loads((runs / 'models' / 'q35-4b' / 'manifest.json').read_text())
    assert manifest['pass1'] == 0.75 and manifest['latency_p50'] == 5.0


def test_template_block_keeps_its_hash_lines():
    """A Qwen3 TEMPLATE contains '# Tools'; only the Modelfile's own comments go."""
    modelfile = ('FROM /abs/x.gguf\nTEMPLATE """{{- if .Tools }}\n# Tools\n\nYou may call'
                 '\n{{ end }}"""\nPARAMETER temperature 0\n# fs2d-assistant trained for x\n')
    assert export.template_lines(modelfile) == ('TEMPLATE """{{- if .Tools }}\n# Tools\n\n'
                                                'You may call\n{{ end }}"""')
