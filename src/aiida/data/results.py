"""AiiDA Data node for one load point's results (spec §4.1).

`FrictionResultsData` stores its results files VERBATIM in the node
repository and re-parses them on demand with `read_data.read_results_file`;
only summary scalars from `RunSummary` are attributes. Time series are NEVER
attributes -- 13k JSONB blobs on SQLite is what this replaces.
"""

import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from aiida.orm import Data  # pyright: ignore[reportMissingImports]

from ..data_keys import TIER_KEYS, content_hash


class FrictionResultsData(Data):
    """One load point: `(load, angle, speed)` for a slide family, one grid for
    a PES scan, one load for an indent.

    The original results files are stored verbatim in the node repository and
    re-parsed on demand with `read_data.read_results_file`; only summary
    scalars are attributes. Time series are NEVER attributes -- 13k JSONB
    blobs on SQLite is what this replaces.
    """

    _IDENTITY = ('family', 'load_kind', 'load', 'angle', 'speed', 'layers',
                 'tiers', 'ntimesteps', 'is_complete', 'content_sha256',
                 'run_uuid', 'set_uuid')

    #: filename prefixes per key. 'indent_' is the retract curve, NOT the window
    #: tier -- filing it under 'window' would make frame('window') return the
    #: force-distance curve for every indent run.
    _TIER_PREFIX = {'window': ('friction_', 'fc_ave_', 'pes_scan'),
                    'raw': ('raw_',), 'peratom': ('peratom_',),
                    'retract': ('indent_',)}

    # `family` is the one validated setter -- it imports the vocabulary rather
    # than repeating a subset of it.
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
    def load_kind(self) -> str:
        return self.base.attributes.get('load_kind', '')

    @load_kind.setter
    def load_kind(self, value: str):
        self.base.attributes.set('load_kind', str(value))

    @property
    def load(self) -> float:
        return self.base.attributes.get('load', 0.0)

    @load.setter
    def load(self, value: float):
        self.base.attributes.set('load', float(value))

    @property
    def angle(self) -> float:
        return self.base.attributes.get('angle', 0.0)

    @angle.setter
    def angle(self, value: float):
        self.base.attributes.set('angle', float(value))

    @property
    def speed(self) -> float:
        return self.base.attributes.get('speed', 0.0)

    @speed.setter
    def speed(self, value: float):
        self.base.attributes.set('speed', float(value))

    @property
    def layers(self) -> Optional[int]:
        return self.base.attributes.get('layers', None)

    @layers.setter
    def layers(self, value: Optional[int]):
        self.base.attributes.set('layers', None if value is None else int(value))

    @property
    def tiers(self) -> List[str]:
        return self.base.attributes.get('tiers', [])

    @tiers.setter
    def tiers(self, value):
        self.base.attributes.set('tiers', [str(x) for x in value])

    @property
    def ntimesteps(self) -> int:
        return self.base.attributes.get('ntimesteps', 0)

    @ntimesteps.setter
    def ntimesteps(self, value: int):
        self.base.attributes.set('ntimesteps', int(value))

    @property
    def is_complete(self) -> bool:
        return self.base.attributes.get('is_complete', True)

    @is_complete.setter
    def is_complete(self, value: bool):
        self.base.attributes.set('is_complete', bool(value))

    @property
    def content_sha256(self) -> str:
        return self.base.attributes.get('content_sha256', '')

    @content_sha256.setter
    def content_sha256(self, value: str):
        self.base.attributes.set('content_sha256', str(value))

    @property
    def run_uuid(self) -> str:
        return self.base.attributes.get('run_uuid', '')

    @run_uuid.setter
    def run_uuid(self, value: str):
        self.base.attributes.set('run_uuid', str(value))

    @property
    def set_uuid(self) -> str:
        return self.base.attributes.get('set_uuid', '')

    @set_uuid.setter
    def set_uuid(self, value: str):
        self.base.attributes.set('set_uuid', str(value))

    # -- Files -----------------------------------------------------------

    @staticmethod
    def _run_relative_name(path: Path) -> str:
        """`<parent-dir-name>/<filename>`.

        R18: every tier and auxiliary file this node stores lives one level
        below the run root (`results/`, `data/`, `lammps/`), so keeping that
        one path component is what lets a future export (Task 17) write a
        member back to the exact run-relative path it came from, instead of
        guessing which subdirectory it belongs in from a bare basename.
        """
        path = Path(path)
        return f'{path.parent.name}/{path.name}'

    def add_result_files(self, paths: Dict[str, Path],
                         aux: Sequence[Path] = (), max_mb: int = 20,
                         content_sha256: str = '') -> None:
        """Store the tier files verbatim, plus logs and hold/final artefacts.

        Each file is stored under its path relative to the run root
        (`results/friction_...txt`, `data/hold_...txt`, `lammps/run_...log`),
        not by bare basename (R18) -- so a future export can write it back to
        the exact run-relative location it came from.

        `content_sha256`, when given, is passed IN, not recomputed: `LoadPoint`
        (Task 15) already hashes every entry of `point.tiers` including
        `'retract'`, and recomputing here over `TIER_KEYS` alone would give a
        different digest whenever an indent retract curve existed, so
        `results_exist` would never match and every re-import of an indent run
        would create duplicates -- the exact opposite of the spec's "second
        ingest creates zero nodes". When no hash is passed (standalone/test
        callers), one is computed here with `content_hash` over the stored
        tier files so the attribute is never empty on a stored node.
        """
        limit = max_mb * 1024 * 1024
        tiers: List[str] = []
        tier_paths: List[Path] = []
        for tier in TIER_KEYS:
            path = paths.get(tier)
            if path is None:
                continue
            path = Path(path)
            self.base.repository.put_object_from_file(
                str(path), self._run_relative_name(path))
            tiers.append(tier)
            tier_paths.append(path)
        for extra in aux:
            extra = Path(extra)
            if extra.is_file() and extra.stat().st_size <= limit:
                self.base.repository.put_object_from_file(
                    str(extra), self._run_relative_name(extra))
        self.tiers = tiers
        self.content_sha256 = content_sha256 or content_hash(tier_paths)

    def set_summary(self, summary) -> None:
        """Write every `RunSummary` scalar as a queryable attribute."""
        for key, val in summary.as_dict().items():
            self.base.attributes.set(key, val)
        self.ntimesteps = int(summary.ntimesteps)
        self.is_complete = bool(summary.is_complete)

    def frame(self, tier: str):
        """Re-parse the stored file for `tier` with the reader's own parser.

        A `.csv` member (a PES grid, an indent retract curve) is a plain comma
        table with a `# family=` first line, not a LAMMPS `ave/time` file, so it
        goes through `pd.read_csv(comment='#')`; everything else goes through
        `read_data.read_results_file`, the same parser the DataReader uses.

        Files are stored under a run-relative path (R18), so the repository is
        walked recursively and matched on the BASENAME's prefix, not the
        top-level object name.
        """
        import pandas as pd  # pylint: disable=import-outside-toplevel

        from ...postprocessing.read_data import read_results_file  # pylint: disable=import-outside-toplevel

        prefixes = self._TIER_PREFIX[tier]
        names = sorted(
            (dirpath / fname).as_posix()
            for dirpath, _dirs, files in self.base.repository.walk()
            for fname in files)
        for name in names:
            basename = Path(name).name
            if not basename.startswith(prefixes):
                continue
            with tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp) / basename
                with self.base.repository.open(name, 'rb') as src:
                    out.write_bytes(src.read())
                if out.suffix == '.csv':
                    return pd.read_csv(out, comment='#')
                # `layers` is already an attribute of this node; without it a
                # four-layer run's v_comx_L3 stays native here and is renamed by
                # the DataReader, so store and reader would disagree about a
                # column present in both. 2 is REGISTRY's declared bilayer set.
                return read_results_file(out, self.family,
                                         n_layers=self.layers or 2)
        raise KeyError(f"no {tier!r} file stored on {self.uuid}")

    def frames_by_tier(self) -> Dict[str, Any]:
        return {tier: self.frame(tier) for tier in self.tiers}

    #: Attribute keys that are NOT part of the summary, so everything else the
    #: results node carries goes into `extras`. An allowlist was tried and
    #: silently dropped `slide_distance`, `driven_com_z_mean`,
    #: `driven_com_z_drift`, `n_points`, `z_span` and `work_of_adhesion_nnA` --
    #: including the three driven-COM values Task 8 describes as the only ones
    #: that separate a real slide from a stalled one. A denylist cannot lose a
    #: future extra by omission; an allowlist always can.
    _NOT_EXTRAS = frozenset(_IDENTITY) | frozenset((
        'mean_cof', 'std_cof', 'mean_lf', 'std_lf', 'mean_nf', 'std_nf',
        'mean_lfx', 'std_lfx', 'mean_lfy', 'std_lfy', 'time_series_hash',
    ))

    def to_record(self, run_node, set_node, *, uploader=None, contact=None):
        """Convert to the central DB's `ResultRecord` (Task 29 defines the
        schema-3 fields `to_record` populates here -- `extras`, `run_uuid`,
        etc. Not exercised by any test until then).
        """
        from ...data.models import ResultRecord  # pylint: disable=import-outside-toplevel

        prov = run_node.provenance or {}
        sheets = [m for m in prov.get('materials', []) if m.get('role') == 'sheet']
        primary = sheets[0] if sheets else {}
        attrs = self.base.attributes
        return ResultRecord(
            uploader=uploader, contact=contact,
            run_uuid=str(run_node.uuid), result_uuid=str(self.uuid),
            set_label=set_node.label, source_host=set_node.source_host,
            model=run_node.model, family=self.family,
            materials=list(run_node.materials),
            material=run_node.materials[0] if run_node.materials else '',
            # No builder records a stacking type on a run, so this stays None
            # and `materials` + `extras['twist_angle']` are the hetero
            # discriminators. Stated here rather than left to be discovered.
            stack_type=None, layers=run_node.layers,
            size_x=run_node.size_x, size_y=run_node.size_y,
            temperature=run_node.temperature,
            tip_material=run_node.tip_material, tip_radius=run_node.tip_radius,
            substrate_material=run_node.substrate,
            substrate_amorphous=run_node.substrate_amorphous,
            potential_type=primary.get('pot_type'),
            potential_name=run_node.pot_token,
            pot_sha256=primary.get('pot_file_sha256'),
            git_commit=prov.get('git_commit'),
            force_nN=self.load if self.load_kind == 'force' else None,
            pressure_gpa=self.load if self.load_kind == 'pressure' else None,
            scan_angle=self.angle, scan_speed=self.speed,
            **{k: attrs.get(k) for k in (
                'mean_cof', 'std_cof', 'mean_lf', 'std_lf', 'mean_nf', 'std_nf',
                'mean_lfx', 'std_lfx', 'mean_lfy', 'std_lfy')},
            ntimesteps=self.ntimesteps,
            time_series_hash=attrs.get('time_series_hash'),
            is_complete=self.is_complete,
            extras=self._extras_for_record(run_node, prov, sheets),
        )

    def _extras_for_record(  # pylint: disable=unused-argument
            self, run_node, prov, sheets) -> Dict[str, Any]:
        """Everything that has no column of its own, plus three provenance keys.

        - every non-identity, non-stat attribute (PES descriptors, indent
          retract features, slip counts, the driven-COM values);
        - `skip_fraction`, so two uploaders with different `settings.database.
          skip_fraction` do not put non-comparable COFs into one table with no
          marker. Attributes are immutable and `results_exist` skips on
          re-ingest, so an unmarked row could never be repaired;
        - `twist_angle` and `potentials`, which are what make a heterostructure
          findable centrally. `potential_type`/`pot_sha256` carry the PRIMARY
          sheet only (they are single-valued columns), so without
          `potentials` a two-material stack loses its second potential -- the
          very single-triple defect the spec's problem table opens with. Two
          hetero runs at 0 deg and 21.8 deg differ in `twist_angle` and in
          nothing else the schema records (`scan_angle` is the slide
          direction, not the stacking twist).

        `run_node` is unused today -- kept in the signature Task 29 will grow
        into, in case a run-level extra is ever needed alongside the
        provenance-derived ones.
        """
        attrs = self.base.attributes
        extras = {k: attrs.get(k) for k in attrs.all
                  if k not in self._NOT_EXTRAS}
        extras['twist_angle'] = prov.get('twist_angle')
        extras['potentials'] = [
            {'name': m.get('name'), 'pot_type': m.get('pot_type'),
             'pot_file_sha256': m.get('pot_file_sha256')}
            for m in sheets]
        return extras

    def __repr__(self) -> str:
        return (f"<FrictionResultsData: {self.family} {self.load_kind}="
                f"{self.load} a{self.angle} s{self.speed} ({self.ntimesteps})>")
