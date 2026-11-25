#!/usr/bin/env python3
"""Write ``manifest.json`` for the materials pack staged in this folder.

Usage::

    python tools/materials_pack/make_manifest.py [FOLDER]

Every ``amor_<mat>__<key8>.lmp`` + ``.lmp.json`` pair at the top level of
FOLDER (default: this script's folder) goes into the manifest with its sha256
and size; ``FrictionSim2D fetch materials`` checks both before it installs
anything. A block without its sidecar, or a sidecar whose ``block_sha256``
disagrees with its block, is refused. README.md, this script, .gitignore and
the manifest itself are not pack files.

Stdlib only, and no frictionsim2d import, so it runs from any Python.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List

SCHEMA = 'materials-pack-1'
MANIFEST = 'manifest.json'
NOT_PACK_FILES = {MANIFEST, 'README.md', 'make_manifest.py', '.gitignore'}


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def problems(folder: Path) -> List[str]:
    """Why the staged files do not make a pack (empty when they do)."""
    found = []
    for block in sorted(folder.glob('amor_*.lmp')):
        sidecar = block.with_name(block.name + '.json')
        if not sidecar.is_file():
            found.append(f'{block.name}: no {sidecar.name} sidecar')
            continue
        try:
            meta = json.loads(sidecar.read_text(encoding='utf-8'))
        except ValueError as exc:
            found.append(f'{sidecar.name}: not JSON ({exc})')
            continue
        recorded = meta.get('block_sha256')
        if recorded and recorded != sha256_of(block):
            found.append(f'{sidecar.name}: block_sha256 does not match {block.name}')
    for sidecar in sorted(folder.glob('amor_*.lmp.json')):
        if not sidecar.with_name(sidecar.name[:-len('.json')]).is_file():
            found.append(f'{sidecar.name}: no block next to it')
    return found


def manifest(folder: Path) -> Dict[str, object]:
    """The manifest for every pack file at the top level of ``folder``."""
    files = {}
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.name not in NOT_PACK_FILES and not path.name.startswith('.'):
            files[path.name] = {'sha256': sha256_of(path), 'size': path.stat().st_size}
    return {'schema': SCHEMA, 'files': files}


def main(argv: List[str] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n', 1)[0])
    parser.add_argument('folder', nargs='?', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args(argv)
    wrong = problems(args.folder)
    if wrong:
        for line in wrong:
            print(f'error: {line}', file=sys.stderr)
        return 1
    data = manifest(args.folder)
    (args.folder / MANIFEST).write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
    print(f"wrote {args.folder / MANIFEST}: {len(data['files'])} file(s)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
