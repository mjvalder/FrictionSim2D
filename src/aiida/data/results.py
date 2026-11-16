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

    @property
    def oversize_files(self) -> List[Dict[str, Any]]:
        """Aux files `add_result_files` recorded by sha256 instead of storing."""
        return self.base.attributes.get('oversize_files', [])

    def add_result_files(self, paths: Dict[str, Path], *, run_dir: Path,
                         aux: Sequence[Path] = (), max_mb: int = 20,
                         content_sha256: str = '') -> None:
        """Store the tier files verbatim, plus logs and hold/final artefacts.

        Each file is stored under its path relative to `run_dir`
        (`results/friction_...txt`, `data/hold_...txt`, `lammps/run_...log`,
        or a bare `log_slide.lammps` sitting directly in the run root), not by
        a `parent-dir-name/basename` guess (R18) -- so a future export can
        write it back to the exact run-relative location it came from. The
        earlier guess broke for any aux file that is a direct child of
        `run_dir` (the slide logs `_aux_files` globs for): its "parent name"
        was the run's own leaf directory, not a real subdirectory, so it was
        stored one level too deep.

        `content_sha256`, when given, is passed IN, not recomputed: `LoadPoint`
        (Task 15) already hashes every entry of `point.tiers` including
        `'retract'`, and recomputing here over `TIER_KEYS` alone would give a
        different digest whenever an indent retract curve existed, so
        `results_exist` would never match and every re-import of an indent run
        would create duplicates -- the exact opposite of the spec's "second
        ingest creates zero nodes". When no hash is passed (standalone/test
        callers), one is computed here with `content_hash` over the stored
        tier files so the attribute is never empty on a stored node.

        An `aux` file over `max_mb` is not stored; its run-relative path,
        size and sha256 are appended to the `oversize_files` attribute instead
        (mirrors `FrictionSimulationData.add_run_files`), so the record still
        says exactly which file was skipped. Tier files are always stored --
        they are the load point's defining data, not a rides-along extra, and
        skipping one would make `frame(tier)` unable to re-parse it.

        Every store goes through `redact.stage_file`: a results node holds no
        settings/config document today, but routing through the same choke
        point as `FrictionSimulationSetData.add_file`/`FrictionSimulationData.
        add_run_files` means one that ever did (a future aux file) would still
        be sanitised, not stored verbatim by omission.
        """
        from ...data.provenance import sha256_file  # pylint: disable=import-outside-toplevel
        from .redact import stage_file  # pylint: disable=import-outside-toplevel

        run_dir = Path(run_dir)
        limit = max_mb * 1024 * 1024
        tiers: List[str] = []
        tier_paths: List[Path] = []
        tier_files: Dict[str, str] = {}
        for tier in TIER_KEYS:
            path = paths.get(tier)
            if path is None:
                continue
            path = Path(path)
            name = path.relative_to(run_dir).as_posix()
            stage_file(self.base.repository, path, name)
            tiers.append(tier)
            tier_paths.append(path)
            tier_files[tier] = name
        oversize: List[Dict[str, Any]] = []
        for extra in aux:
            extra = Path(extra)
            if not extra.is_file():
                continue
            name = extra.relative_to(run_dir).as_posix()
            if extra.stat().st_size > limit:
                oversize.append({'path': name, 'size': extra.stat().st_size,
                                 'sha256': sha256_file(extra),
                                 'source': str(extra)})
                continue
            stage_file(self.base.repository, extra, name)
        self.base.attributes.set('oversize_files', oversize)
        # Which stored member IS each tier: aux may carry other `results/*`
        # files (anything no load point claimed), so `frame` must not guess
        # by filename prefix when it can look the answer up.
        self.base.attributes.set('tier_files', tier_files)
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

        The member is the one `add_result_files` recorded in `tier_files`.
        Nodes without that attribute (and the indent `'retract'` curve, which
        is not a tier) fall back to walking the repository -- files are stored
        under a run-relative path (R18) -- and matching the BASENAME's prefix.
        """
        import pandas as pd  # pylint: disable=import-outside-toplevel

        from ...postprocessing.read_data import read_results_file  # pylint: disable=import-outside-toplevel

        known = self.base.attributes.get('tier_files', None) or {}
        prefixes = self._TIER_PREFIX[tier]
        names = [known[tier]] if tier in known else sorted(
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

    def __repr__(self) -> str:
        return (f"<FrictionResultsData: {self.family} {self.load_kind}="
                f"{self.load} a{self.angle} s{self.speed} ({self.ntimesteps})>")
