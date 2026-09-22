"""The one choke point: no settings/config document enters a friction2d node
repository with a central-DB credential still in it (spec follow-up A).

`redact_secrets` (`src.core.config`) blanks `database.password`/`database.
api_key` at WRITE time, in `src.run.simulate` -- but a tree built before that
existed, or a root-level `settings.yaml` a user wrote by hand (`settings init`
run in the simulation root, which `FrictionSimulationSetData.set_from_root`
also stages), never went through it. `stage_file` is the second, independent
gate: every path staged into a `FrictionSimulationSetData`, `FrictionSimulation
Data` or `FrictionResultsData` repository goes through it instead of calling
`repository.put_object_from_file` directly, so a credential cannot reach the
store, an archive or an export bundle through any of the three node types
regardless of what write-time redaction did or did not do to the source tree.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from ...core.config import redact_secrets_mapping

#: Repository-relative names that are always a settings/config document,
#: regardless of where they sit.
_ALWAYS_DOCUMENT = ('settings.yaml', 'config.json')


def is_settings_or_config_document(name: str) -> bool:
    """True when `name` (a POSIX, repository-relative path) names a
    settings/config document: `settings.yaml`, `config.json`, or any
    `*.yaml`/`*.yml`/`*.json` filed under a `provenance/` prefix.

    `provenance.json` (no `provenance/` prefix -- it sits at the run's own
    root) is deliberately NOT one of these: `src.data.provenance.
    ProvenanceRecord` carries no credential field, so redacting it would only
    cost `FrictionSimulationData.refresh_provenance`'s stored-vs-on-disk sha
    comparison its idempotency for no safety gain.
    """
    if name in _ALWAYS_DOCUMENT:
        return True
    return name.startswith('provenance/') and name.endswith(('.yaml', '.yml', '.json'))


def sanitize_document_bytes(data: bytes, name: str) -> bytes:
    """Redact credentials from one settings/config document's raw bytes.

    `name`'s extension picks the (de)serialiser: `.json` round-trips through
    `json.loads`/`json.dumps`, everything else (`.yaml`/`.yml`) through
    `yaml.safe_load`/`yaml.safe_dump`. The result is re-serialised, not
    byte-identical to the input -- :func:`~src.core.config.
    redact_secrets_mapping` preserves key order and every non-secret value,
    which is what `load_settings` and `rebuild_set` (via a reloaded
    `AFMSimulationConfig`/`SheetOnSheetSimulationConfig`) actually need.
    """
    text = data.decode('utf-8')
    if name.endswith('.json'):
        parsed = json.loads(text)
        redacted: Any = redact_secrets_mapping(parsed)
        return (json.dumps(redacted, indent=2) + '\n').encode('utf-8')
    parsed = yaml.safe_load(text) or {}
    redacted = redact_secrets_mapping(parsed)
    return yaml.safe_dump(redacted, sort_keys=False).encode('utf-8')


def stage_file(repository, source: Path, name: str) -> None:
    """Store `source` under `name` in a node's repository.

    A settings/config document (:func:`is_settings_or_config_document`) is
    sanitised first; every other file is stored verbatim via
    `put_object_from_file`, exactly as every caller did before this module
    existed.
    """
    source = Path(source)
    if not is_settings_or_config_document(name):
        repository.put_object_from_file(str(source), name)
        return
    repository.put_object_from_bytes(
        sanitize_document_bytes(source.read_bytes(), name), name)
