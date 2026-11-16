"""The one choke point: no settings/config document enters a friction2d node
repository with a credential still in it (spec follow-up A).

`redact_secrets` (`src.core.config`) blanks credential-named fields at WRITE
time, in `src.run.simulate` -- but a tree built before that existed, or a
root-level `settings.yaml`/`.yml` a user wrote by hand (`settings init` run in
the simulation root, which `FrictionSimulationSetData.set_from_root` also
stages), never went through it. `stage_file` is the second, independent gate:
every path staged into a `FrictionSimulationSetData`, `FrictionSimulationData`
or `FrictionResultsData` repository -- and, via `src.aiida.submit.
_stage_sim_dir`, a CalcJob's `simulation_dir` input -- goes through it instead
of calling `repository.put_object_from_file` directly, so a credential cannot
reach the store, an archive or an export bundle regardless of what write-time
redaction did or did not do to the source tree.

2026-09-22 review round: the first version of this module keyed candidate
detection off exact names (`settings.yaml`, `config.json`) and a strict
`provenance/` prefix, and `redact_secrets_mapping` (then in `src.core.config`)
keyed redaction off the `database`/`settings` KEY PATH specifically. A probe
got a credential past both: a YAML anchor/alias, `database.extra.password`,
`general.settings.database.password` in a `config.json`, a stray
`hpc.api_key`, and a bare top-level `password` all survived. This version
widens both halves: `is_settings_or_config_document` is now a candidate test
(case-insensitive, name- and location-based, not exact-name), and
`redact_secrets_mapping` (see `src.core.config`) blanks any credential-NAMED
key wherever it occurs, not a fixed set of paths.
"""
from __future__ import annotations

import configparser
import io
import json
from pathlib import Path
from typing import Any

import yaml

from ...core.config import is_secret_key, redact_secrets_mapping

#: Directories immediately under a `provenance` path component whose
#: contents are real structure files (CIFs, potential parameter files) --
#: never settings/config documents, and not worth attempting to parse.
_EXEMPT_PROVENANCE_SUBDIRS = ('cif', 'potentials')


class UnparsableDocumentError(ValueError):
    """A settings/config candidate could not be parsed in any known format.

    Raised instead of storing the file: fails CLOSED. Content nobody has
    actually checked for a credential must never be silently stored just
    because it happens to be malformed YAML/JSON/INI, non-UTF-8, or a
    multi-document YAML stream.
    """


def _is_provenance_candidate(name: str) -> bool:
    """True when `name` sits inside a `provenance` path component, other
    than under its `cif/`/`potentials/` subdirectories.

    The `provenance` component need not be first: `src.aiida.submit.
    _stage_sim_dir` (a CalcJob's `simulation_dir` input) can stage the very
    same file several directories deep, e.g. `afm/h-MoS2/.../K300/provenance/
    settings.yaml` -- `provenance` a middle component, not the first.
    Matching is case-insensitive.
    """
    dirs = [seg.lower() for seg in name.split('/')[:-1]]
    if 'provenance' not in dirs:
        return False
    remainder = dirs[dirs.index('provenance') + 1:]
    return not remainder or remainder[0] not in _EXEMPT_PROVENANCE_SUBDIRS


def _looks_like_settings_or_config_name(basename_lower: str) -> bool:
    """True when `basename_lower` (already lower-cased) names a settings/
    config document, or an obvious variant of one, by its own name alone --
    independent of location: `settings.yaml`/`settings.yml` (plus any
    suffix, e.g. a `.bak` copy), `config.json` (likewise), or any `*.ini`
    file (the run `.ini` a `FrictionSimulationSetData` may also carry, see
    `add_file`).
    """
    if basename_lower.startswith(('settings.yaml', 'settings.yml', 'config.json')):
        return True
    return basename_lower.endswith('.ini')


def is_settings_or_config_document(name: str) -> bool:
    """True when `name` (a POSIX, repository-relative path) is a CANDIDATE
    settings/config document: anything under a `provenance` path component
    (outside `cif/`/`potentials/`), or anything named like one regardless of
    location. Matching is case-insensitive throughout.

    A `True` result commits `stage_file` to CHECKING the file, not
    necessarily to changing it: `sanitize_document_bytes` parses it (JSON,
    then YAML, then -- for a `.ini`-named file -- `configparser`) and raises
    `UnparsableDocumentError` (fails closed) if none of those succeed,
    rather than storing content that was never actually examined.

    `provenance.json` (a `provenance` SIBLING, not a path component under
    it -- it sits at the run's own root) is deliberately never a candidate:
    `src.data.provenance.ProvenanceRecord` carries no credential field, so
    parsing it would only cost `FrictionSimulationData.refresh_provenance`'s
    stored-vs-on-disk sha comparison its idempotency for no safety gain.
    """
    basename_lower = name.rsplit('/', maxsplit=1)[-1].lower()
    return (_is_provenance_candidate(name)
            or _looks_like_settings_or_config_name(basename_lower))


def _sanitize_ini_text(text: str, name: str) -> str:
    """Blank every secret-named option of an INI document, in every section
    (including `DEFAULT`). `configparser` does not preserve comments or
    blank-line formatting -- acceptable here: the only INI this module
    handles today is the run `.ini` a `FrictionSimulationSetData` may carry
    (`add_file`), staged for the record, not reloaded by anything that cares
    about its formatting.
    """
    parser = configparser.ConfigParser(interpolation=None)
    try:
        parser.read_string(text)
    except configparser.Error as exc:
        raise UnparsableDocumentError(
            f"{name}: could not be parsed as INI, and was refused because "
            "its contents could not be checked for credentials before "
            "being stored."
        ) from exc
    for option in list(parser.defaults()):
        if is_secret_key(option):
            parser.set(configparser.DEFAULTSECT, option, '')
    for section in parser.sections():
        for option in parser.options(section):
            if is_secret_key(option):
                parser.set(section, option, '')
    buf = io.StringIO()
    parser.write(buf)
    return buf.getvalue()


def sanitize_document_bytes(data: bytes, name: str) -> bytes:
    """Redact credentials from one settings/config candidate's raw bytes.

    A `.ini`-named file (case-insensitive) goes through `configparser`.
    Everything else tries JSON (`json.loads`) then YAML (`yaml.safe_load` --
    a superset of JSON, so this also catches a JSON document under a
    `.yaml`-ish or extensionless name) in that order; the first that
    succeeds picks the re-serialiser too, so a real `config.json` (valid
    JSON) round-trips as JSON and a real `settings.yaml` as YAML.
    :func:`~src.core.config.redact_secrets_mapping` is applied to whatever
    was parsed -- a mapping, a list, or anything else, it recurses through
    either and passes everything else through unchanged, so this is a no-op
    (beyond re-serialisation) for a document with nothing to redact.

    Not valid UTF-8, or valid in none of INI/JSON/YAML (a multi-document
    YAML stream, for instance) -- either raises `UnparsableDocumentError`:
    fails CLOSED, since content nobody has actually checked for a credential
    must never be silently stored.
    """
    try:
        text = data.decode('utf-8')
    except UnicodeDecodeError as exc:
        raise UnparsableDocumentError(
            f"{name}: not valid UTF-8, and was refused because its contents "
            "could not be checked for credentials before being stored."
        ) from exc

    basename_lower = name.rsplit('/', maxsplit=1)[-1].lower()
    if basename_lower.endswith('.ini'):
        return _sanitize_ini_text(text, name).encode('utf-8')

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        pass
    else:
        redacted: Any = redact_secrets_mapping(parsed)
        return (json.dumps(redacted, indent=2) + '\n').encode('utf-8')

    try:
        parsed = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise UnparsableDocumentError(
            f"{name}: could not be parsed as JSON or YAML, and was refused "
            "because its contents could not be checked for credentials "
            "before being stored."
        ) from exc
    redacted = redact_secrets_mapping(parsed if parsed is not None else {})
    return yaml.safe_dump(redacted, sort_keys=False).encode('utf-8')


def stage_file(repository, source: Path, name: str) -> None:
    """Store `source` under `name` in a node's repository.

    A settings/config candidate (`is_settings_or_config_document`) is
    parsed and sanitised first (`sanitize_document_bytes`, which raises
    `UnparsableDocumentError` rather than store something it could not
    check); every other file is stored verbatim via `put_object_from_file`,
    exactly as every caller did before this module existed.
    """
    source = Path(source)
    if not is_settings_or_config_document(name):
        repository.put_object_from_file(str(source), name)
        return
    repository.put_object_from_bytes(
        sanitize_document_bytes(source.read_bytes(), name), name)
