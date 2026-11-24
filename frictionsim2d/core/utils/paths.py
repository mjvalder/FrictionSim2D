"""Resolving material and potential files to paths on disk.

Config files reference potentials and structures by relative path; these
helpers find them. No structure parsing happens here -- see
:mod:`frictionsim2d.core.utils.structures` for that, and note it is far heavier.
"""
from pathlib import Path
from typing import List, Optional, Union

#: How far above ``base_dir`` to look for a config-relative file. Configs in
#: ``examples/`` name ``examples/materials/...``, i.e. paths relative to the
#: project root rather than to themselves, so the search has to climb.
PATH_SEARCH_MAX_PARENTS = 4


def search_paths(name: str, base_dir: Optional[Union[str, Path]] = None) -> List[Path]:
    """Candidate locations for a config-supplied file, in resolution order.

    Config files reference potentials and structures by relative path, and those
    paths are relative to wherever the user ran from -- in practice the project
    root, not the config's own directory. ``examples/indent_config.ini`` names
    ``examples/materials/h-MoS2.cif``, which resolves one level *above* itself.

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


def find_existing(name: str, base_dir: Optional[Union[str, Path]]) -> Optional[Path]:
    """First existing candidate for ``name``, or None."""
    for candidate in search_paths(name, base_dir):
        if candidate.exists():
            return candidate
    return None


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
        f"Material file '{mat_name}' not found. "
        f"Please provide a valid path to a CIF file."
        f"For a collection of 2D material CIF files, check the 'examples/materials' "
        f"folder in the GitHub repository: https://github.com/Mclat16/FrictionSim2D"
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
        f"Potential file '{pot_name}' not found. "
        f"Please provide a valid path to a potential file. "
        f"For a collection of potential files (sw, tersoff, rebo, reaxff, etc.), "
        f"check the 'examples/potentials' folder in the GitHub repository: "
        f"https://github.com/Mclat16/FrictionSim2D"
    )
