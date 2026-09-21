"""AiiDA Data node for one friction simulation run (spec §4.1).

Every `RunTokens` field (Tasks 3-4) is a queryable attribute; the whole
`ProvenanceRecord` (Task 5) is the `provenance` attribute (a dict); the run's
own inputs and provenance files live in the node repository. Mutable state
(`status`, `imported_at`, `job_id`) lives in EXTRAS, because a stored AiiDA
attribute cannot be changed.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from aiida.orm import Data


def _sanitize(obj: Any) -> Any:
    """Recursively replace float nan/inf with None so AiiDA can store the value.

    This is `src.aiida.integration._sanitize_for_aiida`, moved here as a
    module-level function. `integration.py` keeps its own copy until Task 27
    deletes its half -- it does NOT import this one.
    """
    import math  # pylint: disable=import-outside-toplevel

    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj


class FrictionSimulationData(Data):  # pylint: disable=too-many-public-methods
    """One run: one `provenance.json`.

    Every `RunTokens` field is a queryable attribute; the whole
    `ProvenanceRecord` is the `provenance` attribute (a dict); the run's own
    inputs and provenance files live in the node repository. Mutable state
    (`status`, `imported_at`, `job_id`) lives in EXTRAS, because a stored
    AiiDA attribute cannot be changed.
    """

    STATUSES = ('prepared', 'submitted', 'running',
                'completed', 'failed', 'imported')

    _TOKEN_ATTRS = ('model', 'family', 'materials', 'flake_material',
                    'size_x', 'size_y', 'substrate', 'substrate_amorphous',
                    'tip_material', 'tip_radius', 'temperature', 'layers',
                    'pot_token', 'rel_path')

    @property
    def model(self) -> str:
        return self.base.attributes.get('model', '')

    @model.setter
    def model(self, value: str):
        from ...run.naming import MODELS  # pylint: disable=import-outside-toplevel
        if value not in MODELS:
            raise ValueError(f"model must be one of {MODELS}, got {value!r}")
        self.base.attributes.set('model', str(value))

    @property
    def family(self) -> str:
        return self.base.attributes.get('family', '')

    @family.setter
    def family(self, value: str):
        from ...core.observables import FAMILIES  # pylint: disable=import-outside-toplevel
        if value not in FAMILIES:
            raise ValueError(f"family must be one of {FAMILIES}, got {value!r}")
        self.base.attributes.set('family', str(value))

    @property
    def materials(self) -> List[str]:
        return self.base.attributes.get('materials', [])

    @materials.setter
    def materials(self, value):
        self.base.attributes.set('materials', [str(m) for m in value])

    @property
    def flake_material(self) -> Optional[str]:
        return self.base.attributes.get('flake_material')

    @flake_material.setter
    def flake_material(self, value: Optional[str]):
        self.base.attributes.set('flake_material',
                                 None if value is None else str(value))

    @property
    def size_x(self) -> float:
        return self.base.attributes.get('size_x', 0.0)

    @size_x.setter
    def size_x(self, value: float):
        self.base.attributes.set('size_x', float(value))

    @property
    def size_y(self) -> float:
        return self.base.attributes.get('size_y', 0.0)

    @size_y.setter
    def size_y(self, value: float):
        self.base.attributes.set('size_y', float(value))

    @property
    def substrate(self) -> Optional[str]:
        return self.base.attributes.get('substrate')

    @substrate.setter
    def substrate(self, value: Optional[str]):
        self.base.attributes.set('substrate', None if value is None else str(value))

    @property
    def substrate_amorphous(self) -> Optional[bool]:
        return self.base.attributes.get('substrate_amorphous')

    @substrate_amorphous.setter
    def substrate_amorphous(self, value: Optional[bool]):
        self.base.attributes.set('substrate_amorphous',
                                 None if value is None else bool(value))

    @property
    def tip_material(self) -> Optional[str]:
        return self.base.attributes.get('tip_material')

    @tip_material.setter
    def tip_material(self, value: Optional[str]):
        self.base.attributes.set('tip_material', None if value is None else str(value))

    @property
    def tip_radius(self) -> Optional[float]:
        return self.base.attributes.get('tip_radius')

    @tip_radius.setter
    def tip_radius(self, value: Optional[float]):
        self.base.attributes.set('tip_radius',
                                 None if value is None else float(value))

    @property
    def temperature(self) -> float:
        return self.base.attributes.get('temperature', 300.0)

    @temperature.setter
    def temperature(self, value: float):
        self.base.attributes.set('temperature', float(value))

    @property
    def layers(self) -> Optional[int]:
        return self.base.attributes.get('layers')

    @layers.setter
    def layers(self, value: Optional[int]):
        self.base.attributes.set('layers', None if value is None else int(value))

    @property
    def pot_token(self) -> Optional[str]:
        return self.base.attributes.get('pot_token')

    @pot_token.setter
    def pot_token(self, value: Optional[str]):
        self.base.attributes.set('pot_token', None if value is None else str(value))

    @property
    def rel_path(self) -> str:
        return self.base.attributes.get('rel_path', '')

    @rel_path.setter
    def rel_path(self, value: str):
        self.base.attributes.set('rel_path', str(value))

    @property
    def set_label(self) -> str:
        return self.base.attributes.get('set_label', '')

    @set_label.setter
    def set_label(self, value: str):
        self.base.attributes.set('set_label', str(value))

    @property
    def provenance(self) -> Dict[str, Any]:
        return self.base.attributes.get('provenance', {})

    @provenance.setter
    def provenance(self, value: Dict[str, Any]):
        self.base.attributes.set('provenance', dict(value))

    @property
    def set_uuid(self) -> str:
        return self.base.attributes.get('set_uuid', '')

    @set_uuid.setter
    def set_uuid(self, value: str):
        self.base.attributes.set('set_uuid', str(value))

    @property
    def oversize_files(self) -> List[Dict[str, Any]]:
        """Files `add_run_files` recorded by sha256 instead of storing."""
        return self.base.attributes.get('oversize_files', [])

    # --- mutable state: EXTRAS, because stored attributes are immutable -----

    @property
    def status(self) -> str:
        return self.base.extras.get('status', 'prepared')

    @status.setter
    def status(self, value: str):
        if value not in self.STATUSES:
            raise ValueError(f"status must be one of {self.STATUSES}")
        from datetime import datetime, timezone  # pylint: disable=import-outside-toplevel
        self.base.extras.set('status', value)
        self.base.extras.set('imported_at',
                             datetime.now(timezone.utc).isoformat())

    @property
    def imported_at(self) -> str:
        """When `status` was last set. Empty until the first status write."""
        return self.base.extras.get('imported_at', '')

    @property
    def job_id(self) -> Optional[str]:
        """Scheduler/CalcJob id, written by `submit.submit_simulation`."""
        return self.base.extras.get('job_id')

    @job_id.setter
    def job_id(self, value):
        self.base.extras.set('job_id', None if value is None else str(value))

    # --- population / staging -------------------------------------------

    def set_from_locator(self, locator, set_uuid: str, set_label: str = '') -> None:
        """Populate every attribute from a `RunLocator` and its provenance.

        `set_label` is stored alongside `set_uuid` so a queried run can name its
        set without a second lookup (the query API and `db upload` both need it).
        """
        from ...data.provenance import read_provenance  # pylint: disable=import-outside-toplevel

        tokens = locator.tokens
        self.set_label = str(set_label)
        self.model = tokens.model
        self.family = tokens.family
        self.materials = list(tokens.materials)
        self.flake_material = tokens.flake_material
        self.size_x = float(tokens.size_x)
        self.size_y = float(tokens.size_y)
        self.substrate = tokens.substrate
        self.substrate_amorphous = tokens.substrate_amorphous
        self.tip_material = tokens.tip_material
        self.tip_radius = tokens.tip_radius
        self.temperature = float(tokens.temperature)
        self.layers = tokens.layers
        self.pot_token = tokens.pot_token
        self.rel_path = tokens.rel_path
        self.set_uuid = str(set_uuid)
        record = read_provenance(locator.provenance_file,
                                 family=tokens.family, model=tokens.model,
                                 material_name=tokens.materials[0])
        self.provenance = _sanitize(record.model_dump())

    def add_run_files(self, locator, max_mb: int = 20) -> List[str]:
        """Stage `provenance.json`, `provenance/**`, `lammps/*.in` and
        `lammps/system.in.settings` into the repository.

        A file larger than `max_mb` is not stored; its POSIX path and sha256
        are appended to the `oversize_files` attribute instead, so the record
        still says exactly which file was used.
        """
        from ...data.provenance import sha256_file  # pylint: disable=import-outside-toplevel

        limit = max_mb * 1024 * 1024
        candidates: List[Tuple[Path, str]] = [
            (locator.provenance_file, 'provenance.json')]
        if locator.provenance_dir.is_dir():
            candidates += [(p, 'provenance/' + p.relative_to(
                locator.provenance_dir).as_posix())
                for p in sorted(locator.provenance_dir.rglob('*')) if p.is_file()]
        if locator.lammps_dir.is_dir():
            candidates += [(p, 'lammps/' + p.name)
                           for p in sorted(locator.lammps_dir.glob('*.in'))]
            settings_in = locator.lammps_dir / 'system.in.settings'
            if settings_in.is_file():
                candidates.append((settings_in, 'lammps/system.in.settings'))

        stored: List[str] = []
        oversize: List[Dict[str, Any]] = []
        for path, name in candidates:
            if path.stat().st_size > limit:
                oversize.append({'path': name, 'size': path.stat().st_size,
                                 'sha256': sha256_file(path),
                                 'source': str(path)})
                continue
            self.base.repository.put_object_from_file(str(path), name)
            stored.append(name)
        self.base.attributes.set('oversize_files', oversize)
        return sorted(stored)

    def tokens(self):
        """Rebuild the `RunTokens` this node was created from."""
        from ...run.naming import RunTokens  # pylint: disable=import-outside-toplevel
        return RunTokens(
            model=self.model, family=self.family,
            materials=tuple(self.materials), flake_material=self.flake_material,
            size_x=self.size_x, size_y=self.size_y, substrate=self.substrate,
            substrate_amorphous=self.substrate_amorphous,
            tip_material=self.tip_material, tip_radius=self.tip_radius,
            temperature=self.temperature, layers=self.layers,
            pot_token=self.pot_token, rel_path=self.rel_path)

    def to_dict(self) -> Dict[str, Any]:
        """Every attribute plus the three mutable extras, flat.

        `status`, `imported_at` and `job_id` are EXTRAS, not attributes, but
        they belong in this dict: it is what `QueryResult.to_dataframe` builds
        each row from, and the GUI's run table has an "imported" column with no
        other source for it on a local row. `oversize_files` is NOT in it -- it
        is diagnostic, not a table column.
        """
        out = {'uuid': str(self.uuid), 'pk': self.pk, 'set_uuid': self.set_uuid,
               'set_label': self.set_label, 'status': self.status,
               'imported_at': self.imported_at, 'job_id': self.job_id}
        out.update({k: getattr(self, k) for k in self._TOKEN_ATTRS})
        return out

    def __repr__(self) -> str:
        return f"<FrictionSimulationData: {self.family} {self.rel_path}>"
