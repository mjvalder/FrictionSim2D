"""Building real agent deps over a temporary workspace for tests."""
from pathlib import Path

from src.agent.context import snapshot_deps
from src.agent.sandbox import SearchRoots
from src.gui.controllers.document import ConfigDocument
from src.gui.controllers.settings_document import SettingsDocument

EXAMPLES = Path(__file__).resolve().parents[2] / 'examples'


def make_deps(tmp_path, settings, model='afm', with_examples=True):
    """Deps over ``tmp_path/ws`` with a blank document of ``model``."""
    workspace = tmp_path / 'ws'
    (workspace / 'configs').mkdir(parents=True, exist_ok=True)
    document = ConfigDocument.blank(model)
    document.workspace_dir = workspace / 'configs'
    document.workspace_root = workspace
    roots = SearchRoots(workspace, EXAMPLES if with_examples else None)
    return snapshot_deps(document, SettingsDocument.defaults(), settings, roots)
