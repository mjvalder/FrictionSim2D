"""Building real agent deps over a temporary workspace for tests."""
from pathlib import Path

from frictionsim2d.agent.context import snapshot_deps
from frictionsim2d.agent.sandbox import SearchRoots
from frictionsim2d.gui.controllers.document import ConfigDocument
from frictionsim2d.gui.controllers.settings_document import SettingsDocument

EXAMPLES = Path(__file__).resolve().parents[2] / 'frictionsim2d' / 'data'


def make_deps(tmp_path, settings, model='afm', with_examples=True):
    """Deps over ``tmp_path/ws`` with a blank document of ``model``."""
    workspace = tmp_path / 'ws'
    (workspace / 'configs').mkdir(parents=True, exist_ok=True)
    document = ConfigDocument.blank(model)
    document.workspace_dir = workspace / 'configs'
    document.workspace_root = workspace
    roots = SearchRoots(workspace, EXAMPLES if with_examples else None)
    return snapshot_deps(document, SettingsDocument.defaults(), settings, roots)


CIF_TEXT = 'data_MoS2\n_cell_length_a 3.19\n'


class FakeMaterialsClient:
    """The part of MPClient the materials tools use; never touches the network."""

    def __init__(self, mp_rows=(), dm_rows=(), error=None, cif=CIF_TEXT):
        self.mp_rows, self.dm_rows, self.error = list(mp_rows), list(dm_rows), error
        self.cif = cif
        self.structures = []
        self.downloaded_listing = False

    def _check(self):
        if self.error is not None:
            raise self.error

    def summary(self, formula):  # pylint: disable=unused-argument
        self._check()
        return self.mp_rows

    def twodmatpedia(self, formula):  # pylint: disable=unused-argument
        self._check()
        return self.dm_rows

    def twodmatpedia_entry(self, dm_id):
        self._check()
        return next((r for r in self.dm_rows if r['dm_id'] == dm_id), None)

    def structure_cif(self, structure_id):
        self._check()
        self.structures.append(structure_id)
        return self.cif


def materials_factory(client):
    """An `AgentDeps.materials_db` factory that hands out ``client``."""
    def factory(should_stop=None):
        client.should_stop = should_stop
        return client
    return factory
