"""Installing a fine-tune into Ollama (ollama create from a bundle, or pull)."""
import io

from src.agent.bundles import FineTune, catalogue, write_bundle
from src.agent.ollama_install import install_finetune
from src.agent.settings import AgentSettings

MANIFEST = {'tag': 'q35-4b', 'base': 'unsloth/Qwen3.5-4B', 'schema_hash': 'b8b7',
            'knowledge_hash': 'ea92', 'trained_knowledge': True}


class FakePopen:
    """Records each command; prints scripted output and exits with a scripted code."""

    def __init__(self, scripts=None):
        self.calls = []
        self.scripts = list(scripts or [])

    def __call__(self, cmd, **kwargs):
        self.calls.append((cmd, kwargs))
        out, code = self.scripts.pop(0) if self.scripts else ('success\n', 0)
        process = type('P', (), {})()
        process.stdout = io.StringIO(out)
        process.wait = lambda timeout=None: code
        return process


def _tune(tmp_path):
    gguf = tmp_path / 'w.gguf'
    gguf.write_bytes(b'weights')
    write_bundle(tmp_path / 'models' / 'q35-4b', gguf=gguf, template='RENDERER qwen3.5',
                 manifest=MANIFEST)
    return catalogue(tmp_path / 'models')['fs2d-assistant:q35-4b']


def test_install_from_bundle_runs_create(tmp_path):
    popen = FakePopen()
    tune = _tune(tmp_path)
    ok, message = install_finetune(AgentSettings(), tune, lambda line: None,
                                   popen=popen, binary='/bin/ollama')
    assert ok and message == 'Installed fs2d-assistant:q35-4b'
    cmd, kwargs = popen.calls[0]
    assert cmd == ['/bin/ollama', 'create', 'fs2d-assistant:q35-4b', '-f',
                   str(tune.bundle / 'Modelfile')]
    assert kwargs['env']['OLLAMA_HOST'] == 'localhost:11434'


def test_install_streams_progress(tmp_path):
    lines = []
    popen = FakePopen([('gathering model components\n'
                        'copying file sha256:ab 40%\rcopying file sha256:ab 100%\n'
                        'success\n', 0)])
    install_finetune(AgentSettings(), _tune(tmp_path), lines.append,
                     popen=popen, binary='ollama')
    assert lines == ['checking model.gguf', 'gathering model components',
                     'copying file sha256:ab 40%',
                     'copying file sha256:ab 100%', 'success']


def test_install_refuses_mismatched_gguf(tmp_path):
    popen = FakePopen()
    tune = _tune(tmp_path)
    (tune.bundle / 'model.gguf').write_bytes(b'weig')
    ok, message = install_finetune(AgentSettings(), tune, lambda line: None,
                                   popen=popen, binary='ollama')
    assert not ok and 'does not match' in message and popen.calls == []


def test_install_reports_old_ollama(tmp_path):
    popen = FakePopen([('gathering model components\n'
                        'Error: unknown renderer "qwen3.5"\n', 1)])
    ok, message = install_finetune(AgentSettings(), _tune(tmp_path), lambda line: None,
                                   popen=popen, binary='ollama')
    assert not ok and 'unknown renderer' in message


def test_install_from_registry_pulls_then_copies():
    popen = FakePopen()
    tune = FineTune(tag='x', base='b', schema_hash='s', knowledge_hash='k',
                    trained_knowledge=False, registry='ns/fs2d-assistant:x')
    ok, _ = install_finetune(AgentSettings(), tune, lambda line: None,
                             popen=popen, binary='ollama')
    assert ok
    assert [c for c, _ in popen.calls] == [['ollama', 'pull', 'ns/fs2d-assistant:x'],
                                           ['ollama', 'cp', 'ns/fs2d-assistant:x',
                                            'fs2d-assistant:x']]


def test_failed_pull_skips_copy():
    popen = FakePopen([('Error: pull model manifest: file does not exist\n', 1)])
    tune = FineTune(tag='x', base='b', schema_hash='s', knowledge_hash='k',
                    trained_knowledge=False, registry='ns/fs2d-assistant:x')
    ok, message = install_finetune(AgentSettings(), tune, lambda line: None,
                                   popen=popen, binary='ollama')
    assert not ok and 'does not exist' in message and len(popen.calls) == 1


def test_progress_drops_the_cli_spinner(tmp_path):
    """Ollama redraws a spinner with ANSI codes even into a pipe (0.40.7, seen live)."""
    frame = '\x1b[?2026h\x1b[?25l\x1b[1Ggathering model components {} \x1b[K\x1b[?25h\x1b[?2026l'
    out = (''.join(frame.format(c) for c in '⠙⠹⠸')
           + '\x1b[?2026h\x1b[?25l\x1b[1Gcopying file sha256:f2 100% \x1b[K\n'
           + 'success \x1b[K\x1b[?25h\x1b[?2026l\n')
    lines = []
    install_finetune(AgentSettings(), _tune(tmp_path), lines.append,
                     popen=FakePopen([(out, 0)]), binary='ollama')
    assert lines == ['checking model.gguf', 'gathering model components',
                     'copying file sha256:f2 100%', 'success']
