"""Resolving material and potential files to paths on disk.

Config files reference potentials and structures by relative path; these
helpers find them. No structure parsing happens here -- see
:mod:`frictionsim2d.core.utils.structures` for that, and note it is far heavier.

A name that is not found as given (or next to its config) is looked up in the
libraries, in this order (:func:`library_roots`):

1. ``packaged`` -- ``frictionsim2d/data/{materials,potentials}`` (always there);
2. ``pack`` -- the materials pack, ``$XDG_DATA_HOME/FrictionSim2D/materials-pack``
   (``FrictionSim2D fetch materials``; materials only);
3. ``user`` -- the user library, ``$XDG_DATA_HOME/FrictionSim2D/amorphous``
   (materials only; the amorphous block library writes it);
4. ``legacy`` -- the old repository ``examples/{materials,potentials}`` folders,
   with a deprecation warning.

A library name is ``materials/<file>`` / ``potentials/<dir>/<file>``, or the bare
tail. The old spelling ``examples/materials/...`` still resolves, through the
same roots, with a deprecation warning.
"""
import hashlib
import logging
import os
from pathlib import Path, PurePosixPath
from typing import List, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)

#: How far above ``base_dir`` to look for a config-relative file. Configs may
#: name paths relative to the project root rather than to themselves, so the
#: search has to climb.
PATH_SEARCH_MAX_PARENTS = 4

#: The two library kinds, and the folder each lives in under every root.
LIBRARY_KINDS: Tuple[str, ...] = ('materials', 'potentials')

_PACKAGE_DIR = Path(__file__).resolve().parents[2]
#: The repository's old ``examples/`` folder (exists only in a checkout).
LEGACY_EXAMPLES_DIR = _PACKAGE_DIR.parent / 'examples'

#: Names already warned about, so a sweep logs each legacy path once.
_WARNED: Set[str] = set()


def data_home() -> Path:
    """``$XDG_DATA_HOME`` (default ``~/.local/share``)."""
    return Path(os.environ.get('XDG_DATA_HOME') or str(Path.home() / '.local' / 'share'))


def packaged_dir(kind: str) -> Path:
    """``frictionsim2d/data/<kind>``: shipped with the package, read-only."""
    return _PACKAGE_DIR / 'data' / kind


def materials_pack_dir() -> Path:
    """``$XDG_DATA_HOME/FrictionSim2D/materials-pack``: what ``fetch materials`` installs."""
    return data_home() / 'FrictionSim2D' / 'materials-pack'


def amorphous_library_dir() -> Path:
    """``$XDG_DATA_HOME/FrictionSim2D/amorphous``: the user library, the only
    place amorphous blocks are written."""
    return data_home() / 'FrictionSim2D' / 'amorphous'


def library_roots(kind: str, legacy: bool = True) -> List[Tuple[str, Path]]:
    """``[(label, dir)]`` for ``kind`` in lookup order: packaged, pack, user, legacy.

    The pack and the user library hold materials only. ``legacy=False`` leaves
    out the old ``examples/`` folder.
    """
    if kind not in LIBRARY_KINDS:
        raise ValueError(f"unknown library kind {kind!r}; one of {LIBRARY_KINDS}")
    roots = [('packaged', packaged_dir(kind))]
    if kind == 'materials':
        roots += [('pack', materials_pack_dir()), ('user', amorphous_library_dir())]
    if legacy:
        roots.append(('legacy', LEGACY_EXAMPLES_DIR / kind))
    return roots


def search_paths(name: str, base_dir: Optional[Union[str, Path]] = None) -> List[Path]:
    """Candidate locations for a config-supplied file, in resolution order.

    Config files reference potentials and structures by relative path, and those
    paths are relative to wherever the user ran from -- in practice the project
    root, not the config's own directory: a config in ``runs/`` naming
    ``runs/my.cif`` resolves one level *above* itself.

    So the search tries the name as given (absolute, or relative to the current
    directory), then ``base_dir``, then each of its parents up to
    :data:`PATH_SEARCH_MAX_PARENTS`. That accommodates both conventions without
    asking anyone to rewrite existing configs, and it is what lets the GUI open
    one from an arbitrary working directory.

    Args:
        name: Path or filename from a config.
        base_dir: Directory to resolve relative names against, normally the
            directory containing the config file.

    Returns:
        Candidate paths, most specific first. Never empty.
    """
    candidates = [Path(name)]
    if base_dir is None or Path(name).is_absolute():
        return candidates

    root = Path(base_dir)
    for depth, directory in enumerate([root, *root.parents]):
        if depth > PATH_SEARCH_MAX_PARENTS:
            break
        candidates.append(directory / name)
    return candidates


def _within(path: Path, root: Path) -> Optional[PurePosixPath]:
    """``path`` relative to ``root`` (lexically), or None when outside it."""
    try:
        rel = Path(os.path.normpath(os.path.abspath(path))).relative_to(
            Path(os.path.normpath(os.path.abspath(root))))
    except ValueError:
        return None
    return PurePosixPath(rel.as_posix())


def _installed_tail(path: Path) -> Optional[Tuple[str, PurePosixPath]]:
    """``(kind, tail)`` when ``path`` spells ``.../frictionsim2d/data/<kind>/<tail>``.

    A config written on another machine, or before an environment rebuild,
    holds the packaged library's absolute path there; the same file is in this
    install's library.
    """
    parts = PurePosixPath(path.as_posix()).parts
    for i in range(len(parts) - 3):
        if (parts[i] == 'frictionsim2d' and parts[i + 1] == 'data'
                and parts[i + 2] in LIBRARY_KINDS):
            return parts[i + 2], PurePosixPath(*parts[i + 3:])
    return None


def _library_name(name: str) -> Tuple[Tuple[str, ...], PurePosixPath, bool]:
    """``(kinds, tail, legacy_spelling)``: where in a library root ``name`` points.

    ``examples/materials/x.cif`` (relative, or absolute inside the legacy
    folder) is the legacy spelling; ``materials/x.cif`` names its kind; any
    other relative name is tried under both kinds. An absolute name that spells
    a ``frictionsim2d/data/<kind>/`` path maps onto that kind (another install's
    library); any other absolute name has no library form (``kinds`` is empty).
    """
    path = Path(name)
    if path.is_absolute():
        rel = _within(path, LEGACY_EXAMPLES_DIR)
        if rel is not None and rel.parts and rel.parts[0] in LIBRARY_KINDS:
            return (rel.parts[0],), PurePosixPath(*rel.parts[1:]), True
        installed = _installed_tail(path)
        if installed is not None:
            return (installed[0],), installed[1], False
        return (), PurePosixPath(), False
    parts = PurePosixPath(path.as_posix()).parts
    if len(parts) >= 2 and parts[0] == 'examples' and parts[1] in LIBRARY_KINDS:
        return (parts[1],), PurePosixPath(*parts[2:]), True
    if parts and parts[0] in LIBRARY_KINDS:
        return (parts[0],), PurePosixPath(*parts[1:]), False
    return LIBRARY_KINDS, PurePosixPath(*parts), False


def library_name(path: Union[str, Path]) -> Optional[str]:
    """``materials/<tail>`` / ``potentials/<tail>`` for a path inside any library
    root (packaged, pack, user, legacy), else None.

    The portable spelling of a library file: it resolves to the same file
    (:func:`find_in_library`) on any install, wherever the package lives.
    """
    for kind in LIBRARY_KINDS:
        for _, root in library_roots(kind):
            rel = _within(Path(path), root)
            if rel is not None and rel.parts:
                return f'{kind}/{rel.as_posix()}'
    return None


def _warn_legacy(name: str, found: Path) -> None:
    if name in _WARNED:
        return
    _WARNED.add(name)
    logger.warning(
        "Deprecated path '%s' (the old repository examples/ library), resolved to "
        "%s. Name library files materials/<file> or potentials/<dir>/<file>; "
        "examples/materials and examples/potentials are no longer shipped.", name, found)


def find_in_library(name: str) -> Optional[Path]:
    """``name`` in the library roots (:func:`library_roots`), or None.

    Roots are tried in order (packaged, pack, user, legacy); a name without a
    kind prefix tries materials before potentials within each root. A legacy
    spelling, or a hit in the legacy folder, logs a deprecation warning.
    """
    kinds, tail, legacy_spelling = _library_name(name)
    if not kinds:
        return None
    candidates: List[Tuple[str, Path]] = []
    for kind in kinds:
        candidates += [(label, root / tail) for label, root in library_roots(kind)]
    order = ('packaged', 'pack', 'user', 'legacy')
    candidates.sort(key=lambda c: order.index(c[0]))
    for label, candidate in candidates:
        if candidate.exists():
            if legacy_spelling or label == 'legacy':
                _warn_legacy(name, candidate)
            return candidate
    return None


def _same_bytes(a: Path, b: Path) -> bool:
    if not (a.is_file() and b.is_file()) or a.stat().st_size != b.stat().st_size:
        return a.is_dir() and b.is_dir()
    return _sha256(a) == _sha256(b)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def find_existing(name: str, base_dir: Optional[Union[str, Path]]) -> Optional[Path]:
    """First existing candidate for ``name``, or None.

    The name as given and next to ``base_dir`` (:func:`search_paths`) win, so a
    user's own file is never shadowed by a library one. A hit inside the legacy
    ``examples/`` folder is the one exception: when the library holds the same
    bytes under that name the library copy is returned; a file that differs is
    the user's own and wins, with the deprecation warning.
    """
    legacy_hit = None
    for candidate in search_paths(name, base_dir):
        if candidate.exists():
            if _within(candidate, LEGACY_EXAMPLES_DIR) is not None:
                legacy_hit = legacy_hit or candidate
                continue
            return candidate
    found = find_in_library(name)
    if found is not None and (legacy_hit is None or _same_bytes(found, legacy_hit)):
        return found
    if legacy_hit is not None:
        _warn_legacy(name, legacy_hit)
    return legacy_hit


def _roots_text(kind: str) -> str:
    return ', '.join(str(root) for _, root in library_roots(kind, legacy=False))


def get_material_path(mat_name: str,
                      base_dir: Optional[Union[str, Path]] = None) -> Path:
    """Find material files.

    Args:
        mat_name: Material identifier or file path.
        base_dir: Directory to resolve relative paths against (normally the
            directory holding the config file). Optional; when omitted the
            behaviour is the historical current-working-directory lookup.

    Returns:
        Path to the material file.

    Raises:
        FileNotFoundError: If the material file is not found.
    """
    found = find_existing(mat_name, base_dir)
    if found is not None:
        return found

    raise FileNotFoundError(
        f"Material file '{mat_name}' not found as given, next to the config, or "
        f"in the material library ({_roots_text('materials')}). "
        f"Provide a valid path to a CIF file, or a library name such as "
        f"materials/h-MoS2.cif."
    )


def get_potential_path(pot_name: str,
                       base_dir: Optional[Union[str, Path]] = None) -> Path:
    """Find potential files.

    Args:
        pot_name: Potential name or file path.
        base_dir: Directory to resolve relative paths against (normally the
            directory holding the config file). Optional; when omitted the
            behaviour is the historical current-working-directory lookup.

    Returns:
        Path to the potential file.

    Raises:
        FileNotFoundError: If the potential file is not found.
    """
    found = find_existing(pot_name, base_dir)
    if found is not None:
        return found

    raise FileNotFoundError(
        f"Potential file '{pot_name}' not found as given, next to the config, or "
        f"in the potential library ({_roots_text('potentials')}). "
        f"Provide a valid path to a potential file, or a library name such as "
        f"potentials/sw/MoS2_wen.sw."
    )
