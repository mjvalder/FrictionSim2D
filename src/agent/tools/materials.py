"""Find structures and potentials: the workspace first, then bundled examples."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from ...potentials.utils import EXTENSION_MAP, KNOWN_TYPES
from ..changes import Op, ToolInputError
from ..sandbox import display_path
from . import compact

STRUCTURE_SUFFIX = '.cif'
#: Never potentials, whatever folder they sit in.
_NOT_POTENTIALS = {'.py', '.pyc', '.cif', '.txt', '.md', '.json', '.ini', '.yaml', '.data'}
_SKIP_DIRS = {'__pycache__'}
_FORMULA = re.compile(r'^(?:[A-Z][a-z]?\d*)+$')
MAX_DEPTH = 5
MAX_FILES = 20000


@dataclass(frozen=True)
class FoundFile:
    """A structure or potential file found under one of the search roots."""

    name: str
    kind: str
    pot_type: Optional[str]
    path: Path
    tier: str
    elements: frozenset = frozenset()


@dataclass
class MaterialIndex:
    """Everything found, in tier order (workspace before examples)."""

    structures: List[FoundFile] = field(default_factory=list)
    potentials: List[FoundFile] = field(default_factory=list)
    truncated: List[str] = field(default_factory=list)
    max_files: int = MAX_FILES


def elements_of(name: str) -> frozenset:
    """Element symbols a file or material name encodes; empty when it is not a formula."""
    base = name
    if base.lower().startswith('ffield'):
        base = base.split('.')[-1]
    base = re.sub(r'^[a-z]{1,2}-', '', base)
    base = re.split(r'[_ .]', base)[0]
    if not _FORMULA.match(base):
        return frozenset()
    return frozenset(re.findall(r'[A-Z][a-z]?', base))


def potential_type_of(path: Path) -> Optional[str]:
    """Potential type from the extension, else the folder, else a reax name."""
    suffix = path.suffix.lower()
    if suffix in _NOT_POTENTIALS:
        return None
    if suffix in EXTENSION_MAP:
        return EXTENSION_MAP[suffix]
    parent = path.parent.name.lower()
    if parent in KNOWN_TYPES:
        return parent
    if 'reax' in path.name.lower():
        return 'reaxff'
    return None


def _stem(path: Path) -> str:
    return path.name.split('.')[0] if path.name.lower().startswith('ffield') else path.stem


def build_index(roots, max_files: int = MAX_FILES) -> MaterialIndex:
    """Walk each root (depth- and count-limited per tier), skipping dot-folders."""
    index = MaterialIndex(max_files=max_files)
    for tier, root in roots.tiers():
        root = Path(root)
        seen = 0
        full = False
        for current, dirs, files in os.walk(root):
            depth = len(Path(current).relative_to(root).parts)
            dirs[:] = sorted(d for d in dirs if not d.startswith('.') and d not in _SKIP_DIRS
                             and depth < MAX_DEPTH)
            for filename in sorted(files):
                path = Path(current) / filename
                if path.suffix.lower() == STRUCTURE_SUFFIX:
                    found = FoundFile(path.stem, 'structure', None, path, tier,
                                      elements_of(path.stem))
                    target = index.structures
                else:
                    pot_type = potential_type_of(path)
                    if not pot_type:
                        continue
                    found = FoundFile(_stem(path), 'potential', pot_type, path, tier,
                                      elements_of(path.name))
                    target = index.potentials
                seen += 1
                if seen > max_files:
                    full = True
                    break
                target.append(found)
            if full:
                index.truncated.append(tier)
                break
    return index


def index_for(deps) -> MaterialIndex:
    """The index, built once per turn and cached on the deps."""
    if deps.material_index is None:
        deps.material_index = build_index(deps.roots)
    return deps.material_index


def _norm(text: str) -> str:
    return re.sub(r'[\s_\-]', '', text).casefold()


def _note(deps) -> str:
    index = index_for(deps)
    return ''.join(f'\n({tier} search stopped after {index.max_files} files)'
                   for tier in index.truncated)


def search_structures(deps, query: str) -> str:
    """Structure files whose name contains the query (case and dashes ignored)."""
    wanted = _norm(query)
    hits = [f for f in index_for(deps).structures if wanted in _norm(f.name)]
    if not hits:
        return f"no structure matching '{query}' in the workspace or examples" + _note(deps)
    return compact('\n'.join(f'{f.name} -- {display_path(f.path, deps.roots)}'
                             for f in hits[:15])) + _note(deps)


def _potentials_for(deps, material: str, potential_type: Optional[str]) -> List[FoundFile]:
    wanted = elements_of(material)
    exact = [f for f in index_for(deps).potentials if f.name.casefold() == material.casefold()]
    by_elements = [f for f in index_for(deps).potentials
                   if wanted and f.elements == wanted and f not in exact]
    found = exact + by_elements
    if potential_type:
        found = [f for f in found if f.pot_type == potential_type.lower()]
    return found


def search_potentials(deps, material: str, potential_type: Optional[str] = None) -> str:
    """Potential files for a material (exact name or same elements), optionally by type."""
    hits = _potentials_for(deps, material, potential_type)
    if not hits:
        kind = f'{potential_type} ' if potential_type else ''
        return f"no {kind}potential for '{material}' in the workspace or examples" + _note(deps)
    return compact('\n'.join(f'{f.pot_type}: {f.path.name} -- {display_path(f.path, deps.roots)}'
                             for f in hits[:15])) + _note(deps)


def resolve_material(deps, name: str,
                     potential_type: Optional[str] = None) -> Tuple[FoundFile, FoundFile]:
    """The first structure and potential for ``name`` (exact stem preferred)."""
    structures = [f for f in index_for(deps).structures if f.name.casefold() == name.casefold()]
    if not structures:
        raise ToolInputError(f"no structure file named '{name}'; call search_structures "
                             'and use a name exactly as it returns')
    potentials = _potentials_for(deps, name, potential_type)
    if not potentials:
        kind = f'{potential_type} ' if potential_type else ''
        raise ToolInputError(f"no {kind}potential for '{name}'; call search_potentials "
                             'or ask the user which potential to use')
    return structures[0], potentials[0]


#: In-plane angles further apart than this are different lattices.
GAMMA_TOL_DEG = 1.0


def _structure(deps, name: str) -> FoundFile:
    """The structure file named exactly ``name``."""
    for found in index_for(deps).structures:
        if found.name.casefold() == name.casefold():
            return found
    raise ToolInputError(f"no structure file named '{name}'; call search_structures "
                         'and use a name exactly as it returns')


def _cell_line(name: str, cell: dict) -> str:
    return (f"{name}: a {cell['lat_a']:.3f}, b {cell['lat_b']:.3f} Angstrom, "
            f"gamma {cell['ang_g']:.1f} deg")


def _percent(x: float, y: float) -> float:
    return abs(x - y) / ((x + y) / 2) * 100


def lattice_mismatch(deps, material_a: str, material_b: Optional[str] = None) -> str:
    """In-plane cell of one material; for two, their mismatch and smallest coincidence cell."""
    # pylint: disable=import-outside-toplevel
    from ...core.config import LatticeMatchConfig
    from ...core.utils.structures import cifread  # reaches ASE (~270 ms): only when called
    from ...lattice.matching import cell_from_params, find_coincidence_lattice
    found_a = _structure(deps, material_a)
    cell_a = cifread(found_a.path)
    source_a = f'(from {display_path(found_a.path, deps.roots)})'
    if material_b is None:
        return compact(f'{_cell_line(found_a.name, cell_a)} {source_a}')
    found_b = _structure(deps, material_b)
    cell_b = cifread(found_b.path)
    lines = [f"{found_a.name} / {found_b.name}: in-plane mismatch "
             f"a {_percent(cell_a['lat_a'], cell_b['lat_a']):.2f} %, "
             f"b {_percent(cell_a['lat_b'], cell_b['lat_b']):.2f} % (|x1 - x2| / mean)",
             _cell_line(found_a.name, cell_a), _cell_line(found_b.name, cell_b)]
    if abs(cell_a['ang_g'] - cell_b['ang_g']) > GAMMA_TOL_DEG:
        lines.append('different in-plane lattice (gamma differs): the a/b mismatch does '
                     'not describe the interface; the coincidence cell does')
    # The hetero builder's own defaults, so this never contradicts what it builds
    # (10 repeats: ~2 s for a pair with no match).
    search = LatticeMatchConfig()
    match = find_coincidence_lattice(
        cell_from_params(cell_a['lat_a'], cell_a['lat_b'], cell_a['ang_g']),
        cell_from_params(cell_b['lat_a'], cell_b['lat_b'], cell_b['ang_g']),
        strain_tol=search.strain_tol, max_supercell=search.max_supercell,
        area_tol=search.area_tol)
    bounds = (f'<= {search.max_supercell} repeats a side, strain <= '
              f'{search.strain_tol * 100:g} %, as the hetero builder searches')
    if match is None:
        lines.append(f'no coincidence cell within {bounds}')
    else:
        lines.append(f'smallest coincidence cell ({bounds}): '
                     f'{found_a.name} {match.matrix_a.tolist()}, '
                     f'{found_b.name} {match.matrix_b.tolist()}, area {match.area:.2f} A^2, '
                     f'strain {match.strain_a * 100:.2f} % / {match.strain_b * 100:.2f} %')
    lines.append(f'files: {display_path(found_a.path, deps.roots)}, '
                 f'{display_path(found_b.path, deps.roots)}')
    return compact('\n'.join(lines))


def config_path(deps, path: Path) -> str:
    """``path`` as a config value: relative to the config's folder when it can be."""
    base = deps.document.base_dir
    if base is not None:
        try:
            return str(Path(path).relative_to(Path(base)))
        except ValueError:
            pass
    return str(path)


def material_ops(deps, names: List[str],
                 potential_type: Optional[str] = None) -> Tuple[List[Op], List[str]]:
    """Operations that point the [2D] sheet at one material, or sweep several."""
    resolved = [resolve_material(deps, n, potential_type) for n in names]
    notes = [f'{n}: {display_path(s.path, deps.roots)} + {p.pot_type} '
             f'{display_path(p.path, deps.roots)}' for n, (s, p) in zip(names, resolved)]
    if len(names) == 1:
        structure, potential = resolved[0]
        ops = [Op('set', '2D', 'mat', names[0]),
               Op('set', '2D', 'cif_path', config_path(deps, structure.path)),
               Op('set', '2D', 'pot_path', config_path(deps, potential.path)),
               Op('set', '2D', 'pot_type', potential.pot_type)]
        if deps.document.has_material_list():
            ops.insert(0, Op('materials', value=[]))
        return ops, notes
    structure_dirs = {s.path.parent for s, _ in resolved}
    potential_dirs = {p.path.parent for _, p in resolved}
    suffixes = {p.path.suffix for _, p in resolved}
    named_after = all(p.name == n for n, (_, p) in zip(names, resolved))
    if not named_after or any(len(group) != 1
                              for group in (structure_dirs, potential_dirs, suffixes)):
        raise ToolInputError(
            'a material sweep needs one structure folder and one potential folder whose '
            'files are named after the materials (e.g. examples/materials with '
            'examples/potentials/sw/sw_lammps); set up one material at a time or ask '
            'the user. Found: ' + '; '.join(notes))
    ops = [Op('material_dirs', value=(str(structure_dirs.pop()), str(potential_dirs.pop()),
                                       suffixes.pop())),
           Op('materials', value=list(names)),
           Op('set', '2D', 'pot_type', resolved[0][1].pot_type)]
    return ops, notes
