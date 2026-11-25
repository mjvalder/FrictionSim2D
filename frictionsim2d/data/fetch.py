"""The materials pack: pre-quenched amorphous blocks, fetched on the user's request.

The pack is a Hugging Face dataset repo (:data:`PACK_REPO`) holding flat
``amor_<mat>__<key8>.lmp`` + ``.lmp.json`` files (the user-library format, so a
user block uploads as is) and a ``manifest.json`` naming every file with its
sha256 and size (written by ``tools/materials_pack/make_manifest.py``).

:func:`fetch_materials` downloads the manifest and every file into a staging
folder next to the pack, checks each size and sha256, and only then swaps the
staging folder in for :func:`~frictionsim2d.core.utils.paths.materials_pack_dir`.
A failed or interrupted fetch leaves the installed pack as it was, so a partial
or corrupt download is never on the lookup path.

``huggingface_hub`` is used when it is installed (the ``agent`` extra has it);
otherwise files come over plain HTTPS from the repo's ``resolve/main`` URLs, so
no extra is needed.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import urllib.request
from pathlib import Path, PurePosixPath
from typing import Callable, Dict, Optional
from urllib.parse import quote

from ..core.utils.paths import materials_pack_dir

#: The dataset repo the pack is published to.
PACK_REPO = 'mjvalder/fs2d-materials'
MANIFEST_NAME = 'manifest.json'
MANIFEST_SCHEMA = 'materials-pack-1'

#: ``download(filename, folder) -> Path``: fetch one repo file into ``folder``.
Download = Callable[[str, Path], Path]


class FetchError(RuntimeError):
    """The pack could not be fetched or verified; the message says why."""


def sha256_of(path: Path) -> str:
    """The hex sha256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(path: Path) -> Dict[str, Dict[str, object]]:
    """``{file: {sha256, size}}`` from a pack manifest.

    Raises:
        FetchError: not JSON, the wrong schema, or a file entry that is not a
            flat name with a sha256 and a size.
    """
    try:
        data = json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        raise FetchError(f'unreadable pack manifest {path}: {exc}') from exc
    if not isinstance(data, dict) or data.get('schema') != MANIFEST_SCHEMA:
        raise FetchError(f'{path} is not a {MANIFEST_SCHEMA} manifest')
    files = data.get('files')
    if not isinstance(files, dict):
        raise FetchError(f'{path} has no "files" table')
    for name, entry in files.items():
        pure = PurePosixPath(name)
        if (len(pure.parts) != 1 or name in ('.', '..', MANIFEST_NAME)
                or '\\' in name):
            raise FetchError(f'pack manifest names {name!r}: files must be flat names')
        if (not isinstance(entry, dict) or not isinstance(entry.get('sha256'), str)
                or not isinstance(entry.get('size'), int)):
            raise FetchError(f'pack manifest entry {name!r} needs a sha256 and a size')
    return files


def materials_available(pack_dir: Optional[Path] = None) -> bool:
    """Whether a complete pack is installed: its manifest reads and every file it
    lists is there at the listed size. (Checksums are checked at fetch time.)"""
    folder = Path(pack_dir) if pack_dir is not None else materials_pack_dir()
    try:
        files = read_manifest(folder / MANIFEST_NAME)
    except FetchError:
        return False
    for name, entry in files.items():
        path = folder / name
        if not path.is_file() or path.stat().st_size != entry['size']:
            return False
    return True


def _https_download(repo: str) -> Download:
    def download(filename: str, folder: Path) -> Path:
        url = f'https://huggingface.co/datasets/{repo}/resolve/main/{quote(filename)}'
        dest = folder / filename
        with urllib.request.urlopen(url, timeout=60) as response:  # an https URL only
            with open(dest, 'wb') as out:
                shutil.copyfileobj(response, out)
        return dest
    return download


def default_download(repo: str = PACK_REPO) -> Download:
    """``huggingface_hub.hf_hub_download`` when installed, else plain HTTPS."""
    try:
        from huggingface_hub import hf_hub_download  # pylint: disable=import-outside-toplevel
    except ImportError:
        return _https_download(repo)

    def download(filename: str, folder: Path) -> Path:
        return Path(hf_hub_download(repo_id=repo, repo_type='dataset', filename=filename,
                                    local_dir=str(folder)))
    return download


def _verified(path: Path, name: str, entry: Dict[str, object]) -> Optional[str]:
    """None when ``path`` matches ``entry``, else what is wrong."""
    if not path.is_file():
        return f'{name}: not downloaded'
    if path.stat().st_size != entry['size']:
        return f"{name}: {path.stat().st_size} bytes, the manifest says {entry['size']}"
    if sha256_of(path) != entry['sha256']:
        return f'{name}: sha256 does not match the manifest'
    return None


def _swap_in(staging: Path, target: Path) -> None:
    """Replace ``target`` by ``staging`` with two renames (same parent folder)."""
    old = target.with_name(target.name + '.old')
    shutil.rmtree(old, ignore_errors=True)
    if target.exists():
        target.rename(old)
    staging.rename(target)
    shutil.rmtree(old, ignore_errors=True)


def fetch_materials(progress: Callable[[str], None], *,
                    download: Optional[Download] = None,
                    pack_dir: Optional[Path] = None,
                    repo: str = PACK_REPO) -> Path:
    """Install (or refresh) the materials pack; the pack folder.

    Files already installed with the manifest's sha256 are reused, not
    downloaded again. Nothing at ``pack_dir`` changes unless every file
    verifies.

    Raises:
        FetchError: a download failed, the manifest is bad, or a file does not
            match it.
    """
    target = Path(pack_dir) if pack_dir is not None else materials_pack_dir()
    download = download or default_download(repo)
    staging = target.with_name(target.name + '.partial')
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    try:
        progress(f'downloading {MANIFEST_NAME} from huggingface.co/datasets/{repo}')
        files = read_manifest(download(MANIFEST_NAME, staging))
        for name, entry in files.items():
            dest = staging / name
            installed = target / name
            if _verified(installed, name, entry) is None:
                shutil.copy2(installed, dest)
                continue
            progress(f'downloading {name}')
            got = Path(download(name, staging))
            if got != dest:
                shutil.move(str(got), str(dest))
            problem = _verified(dest, name, entry)
            if problem:
                raise FetchError(problem)
        # hf_hub_download(local_dir=...) keeps its download metadata in
        # <local_dir>/.cache; it is not part of the pack.
        shutil.rmtree(staging / '.cache', ignore_errors=True)
        _swap_in(staging, target)
    except FetchError:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    except Exception as exc:
        shutil.rmtree(staging, ignore_errors=True)
        raise FetchError(f'download from huggingface.co/datasets/{repo} failed: {exc}') from exc
    progress(f'{len(files)} file(s) verified')
    return target


def remove_materials(pack_dir: Optional[Path] = None) -> bool:
    """Delete the installed pack; whether there was one."""
    target = Path(pack_dir) if pack_dir is not None else materials_pack_dir()
    existed = target.exists()
    shutil.rmtree(target, ignore_errors=True)
    return existed
