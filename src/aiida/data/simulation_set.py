"""AiiDA Data node representing a batch (set) of friction simulations.

A simulation set corresponds to one ``simulation_XXXXXXXX`` output folder
produced by a single invocation of ``FrictionSim2D run``. It carries the
shared identity of that campaign -- its label, model, the tree it was built
from, and its materials -- as queryable attributes; ``settings.yaml`` and the
run ``.ini`` live in the node repository instead.

Every ``FrictionSimulationData`` and ``FrictionResultsData`` node in the same
batch carries a ``set_uuid`` attribute pointing back to this node -- runs
point up, this node never points down (there is no ``run_uuids`` property).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from aiida.orm import Data


class FrictionSimulationSetData(Data):
    """One `simulation_*` root: the shared identity of a campaign.

    Attributes are the queryable half; `settings.yaml` and the run `.ini` live
    in the node repository. There is no SIMULATION_TYPES tuple any more -- the
    vocabulary is `src.core.observables.FAMILIES` and `run.naming.MODELS`.
    """

    _ATTRS = ('label', 'model', 'source_root', 'created', 'git_commit',
              'git_dirty', 'materials', 'n_runs', 'source_host')

    @property
    def label(self) -> str:
        return self.base.attributes.get('label', '')

    @label.setter
    def label(self, value: str):
        cleaned = str(value or '').strip()
        if not cleaned:
            raise ValueError("label is required and cannot be empty")
        if not self.is_stored:
            self.base.attributes.set('label', cleaned)
        from aiida.orm import Node as _Node  # pylint: disable=import-outside-toplevel
        _Node.label.fset(self, cleaned)

    # Seven more attributes follow the same two-line getter/setter shape, one
    # pair each, with the coercion and default given in this table:
    #
    #   name          type        default   setter stores
    #   ------------- ----------- --------- --------------------------------------
    #   model         str         ''        str(value)
    #   source_root   str         ''        str(value)   (absolute, resolved)
    #   created       str         ''        str(value)          (ISO-8601 UTC)
    #   git_commit    str         ''        str(value)
    #   git_dirty     bool        False     bool(value)
    #   materials     List[str]   []        sorted(set(str(m) for m in value))
    #   n_runs        int         0         int(value)
    #   source_host   str         ''        str(value)
    #
    # written out, that shape is:
    @property
    def model(self) -> str:
        return self.base.attributes.get('model', '')

    @model.setter
    def model(self, value: str):
        # The vocabulary comes from run.naming, not a private tuple here: that
        # is exactly the drift SIMULATION_TYPES = ('afm', 'sheetonsheet') caused.
        from ...run.naming import MODELS  # pylint: disable=import-outside-toplevel
        if value not in MODELS:
            raise ValueError(f"model must be one of {MODELS}, got {value!r}")
        self.base.attributes.set('model', str(value))

    @property
    def source_root(self) -> str:
        return self.base.attributes.get('source_root', '')

    @source_root.setter
    def source_root(self, value: str):
        self.base.attributes.set('source_root', str(value))

    @property
    def created(self) -> str:
        return self.base.attributes.get('created', '')

    @created.setter
    def created(self, value: str):
        self.base.attributes.set('created', str(value))

    @property
    def git_commit(self) -> str:
        return self.base.attributes.get('git_commit', '')

    @git_commit.setter
    def git_commit(self, value: str):
        self.base.attributes.set('git_commit', str(value))

    @property
    def git_dirty(self) -> bool:
        return self.base.attributes.get('git_dirty', False)

    @git_dirty.setter
    def git_dirty(self, value: bool):
        self.base.attributes.set('git_dirty', bool(value))

    @property
    def materials(self) -> List[str]:
        return self.base.attributes.get('materials', [])

    @materials.setter
    def materials(self, value: List[str]):
        self.base.attributes.set('materials', sorted(set(str(m) for m in value)))

    @property
    def n_runs(self) -> int:
        return self.base.attributes.get('n_runs', 0)

    @n_runs.setter
    def n_runs(self, value: int):
        self.base.attributes.set('n_runs', int(value))

    @property
    def source_host(self) -> str:
        return self.base.attributes.get('source_host', '')

    @source_host.setter
    def source_host(self, value: str):
        self.base.attributes.set('source_host', str(value))

    # -- Factory / population -------------------------------------------------

    def set_from_root(self, root: Path, label: str) -> None:
        """Populate from a built simulation root and stage its settings.yaml."""
        import socket  # pylint: disable=import-outside-toplevel
        from datetime import datetime, timezone  # pylint: disable=import-outside-toplevel

        from ...run.tree import discover_runs  # pylint: disable=import-outside-toplevel
        from ...data.provenance import read_provenance  # pylint: disable=import-outside-toplevel

        root = Path(root)
        locators = discover_runs(root)
        if not locators:
            raise ValueError(f"no runs found under {root}")

        self.label = label
        self.model = locators[0].tokens.model
        # the tree this set was built from; `find_or_create_set` refuses a
        # second, different tree under the same label
        self.source_root = str(Path(root).resolve())
        self.created = datetime.now(timezone.utc).isoformat()
        self.source_host = socket.gethostname()
        self.n_runs = len(locators)
        self.materials = sorted({m for loc in locators
                                 for m in loc.tokens.materials})

        first = read_provenance(locators[0].provenance_file,
                                family=locators[0].tokens.family,
                                model=locators[0].tokens.model)
        self.git_commit = first.git_commit
        self.git_dirty = bool(first.git_dirty)

        settings = root / 'settings.yaml'
        if not settings.is_file():
            settings = locators[0].provenance_dir / 'settings.yaml'
        if settings.is_file():
            self.add_file(settings, 'settings.yaml')

    def add_file(self, path: Path, name: str) -> None:
        """Stage one file into the node repository under `name`."""
        self.base.repository.put_object_from_file(str(Path(path)), name)

    # -- Serialisation --------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        out = {'uuid': str(self.uuid), 'pk': self.pk}
        out.update({key: getattr(self, key) for key in self._ATTRS})
        return out

    def __repr__(self) -> str:
        return (f"<FrictionSimulationSetData: '{self.label}' "
                f"{self.model} ({self.n_runs} runs)>")
