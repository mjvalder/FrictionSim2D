"""Retention verification (spec §5 F6).

Every artefact declared in ``provenance.json`` must exist with a matching
checksum. A LAYER2-style post-hoc reconstruction should succeed on 100 % of
runs, not 1220/1248.
"""
from pathlib import Path
from typing import List

from ..data.provenance import sha256_file


def verify_retention(run_dir: Path, provenance: dict) -> List[str]:
    """Return the list of missing or checksum-mismatched artefacts.

    Empty list = pass. Paths in the provenance record are relative to
    ``run_dir``.
    """
    run_dir = Path(run_dir)
    problems: List[str] = []
    for art in provenance.get('retained_artefacts', []):
        rel = art['path']
        target = run_dir / rel
        if not target.is_file():
            problems.append(rel)
            continue
        if sha256_file(target) != art['sha256']:
            problems.append(f'{rel} (checksum mismatch)')
    return problems


def register_artefacts(run_dir: Path, patterns: List[str]) -> List[dict]:
    """Checksum every file matching the given glob patterns under run_dir.

    Returns entries in the ProvenanceRecord.retained_artefacts shape.
    """
    run_dir = Path(run_dir)
    entries = []
    for pattern in patterns:
        for f in sorted(run_dir.glob(pattern)):
            if f.is_file():
                entries.append({
                    'path': str(f.relative_to(run_dir)),
                    'sha256': sha256_file(f),
                })
    return entries
