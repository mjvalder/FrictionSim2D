"""Every shipped ``examples/*.ini`` through parse -> expand -> build -> render.

Spec Phase 3, item 1. ``tests/run/test_orchestration.py`` already generates four
of the ten and checks that ``read_data`` paths resolve from the simulation root;
this module covers all ten -- including the two campaign files no test read
before -- and asserts the four contracts that go unchecked today:

1. every atom type a ``pair_coeff`` names exists in the built system,
2. every group a ``fix``/``compute``/``velocity``/``dump`` names is defined,
3. the box the script creates matches the ``[2D] x``/``y`` that was asked for,
4. every potential file a ``pair_coeff`` names resolves from the simulation root.

Nothing is mocked. Builds are session-scoped and shared: ``generate_package`` is
the slow part, so each config is built at most once per session and every test
reads that one tree. No build ever melt-quenches; the shipped
``frictionsim2d/data/materials/amor_Si__021d81ab.lmp`` block is what they reuse.

Two families need special handling.

*Campaigns.* ``indent_campaign.ini`` and ``afm_freestanding_pes_campaign.ini``
expand to 156 members each -- 98 s and 31 s to build in full, which would nearly
double the fast tier for nothing, since both are pure material sweeps and every
member renders the same script shape. Per the Phase 3 plan's decision D4 the
member count is asserted with :func:`count_expansion` against the real file (no
build at all), and the artifact assertions run against a copy whose
``materials_list`` names one material. The copy differs from the original only in
that list, so its first member is byte-for-byte the member the real campaign
would build first.

*Hetero.* A hetero build shells out to ``lmp_serial``
(``frictionsim2d/builders/construction/hetero_stack.py`` ``run_lammps``), so it cannot run
in the LAMMPS-free fast tier. Its case carries an ``lmp_serial`` skipif and
``conftest.py`` derives the ``lammps`` marker from that guard automatically.
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pytest

from frictionsim2d.core.config import read_config
from frictionsim2d.run.expand import count_expansion
from tests.helpers.examples import (
    CAMPAIGNS, CASES, EXAMPLES, HETERO, REPO_ROOT,
)

pytestmark = pytest.mark.integration

LMP = shutil.which('lmp_serial')

ALL_BUILT = CASES + [(n, m, 1) for n, m, _ in CAMPAIGNS]

#: LAMMPS commands that take a group-ID, and which argument it is.
#: ``fix``/``compute``/``dump`` are ``<cmd> ID group-ID style ...``; ``velocity``
#: is ``velocity group-ID style ...`` with no ID -- reading it at index 2 picks
#: up the style ("zero", "create", "set") and reports a group that never existed.
_GROUP_ARG = {'fix': 2, 'compute': 2, 'dump': 2, 'velocity': 1}

_CREATE_BOX_RE = re.compile(r'^\s*create_box\s+(\d+)\s', re.M)
_REGION_BOX_RE = re.compile(
    r'^\s*region\s+\S+\s+block\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s', re.M)
_ATOM_TYPES_RE = re.compile(r'^\s*(\d+)\s+atom types\s*$', re.M)
_DATA_BOUNDS_RE = re.compile(r'^\s*(\S+)\s+(\S+)\s+([xy])lo \3hi\s*$', re.M)
_INCLUDE_RE = re.compile(r'^\s*include\s+(\S+)', re.M)
_READ_DATA_RE = re.compile(r'^\s*read_data\s+(\S+)([^\n]*)', re.M)
_EXTRA_TYPES_RE = re.compile(r'\bextra/atom/types\s+(\d+)')
_GROUP_DEF_RE = re.compile(r'^\s*group\s+(\S+)', re.M)
_READ_DATA_GROUP_RE = re.compile(r'^\s*read_data\s+.*?\bgroup\s+(\S+)', re.M)
_PAIR_COEFF_RE = re.compile(r'^\s*pair_coeff\s+(\S+)\s+(\S+)\s+(.*?)\s*(?:#.*)?$',
                            re.M)


# --------------------------------------------------------------------------
# Reading a built member
# --------------------------------------------------------------------------

def _resolve(root: Path, ref: str) -> Optional[Path]:
    """A script-internal path, resolved the way LAMMPS will resolve it.

    Every baked path is relative to the *simulation root* -- the directory the
    job is launched from -- not to the script's own directory. Returns None for
    an unexpanded LAMMPS variable, which is not a file reference.
    """
    if '$' in ref:
        return None
    return root / ref.strip('"\'')


def _script_text(root: Path, script: Path) -> str:
    """The script plus every file it ``include``s, concatenated.

    ``system.in.settings`` holds the masses, the component groups and every
    ``pair_coeff``; reading the script alone would see fixes referring to groups
    that look undefined.
    """
    parts = [script.read_text(encoding='utf-8')]
    for ref in _INCLUDE_RE.findall(parts[0]):
        target = _resolve(root, ref)
        if target is not None and target.is_file():
            parts.append(target.read_text(encoding='utf-8'))
    return '\n'.join(parts)


def _atom_type_count(root: Path, texts: List[str]) -> int:
    """How many atom types the built system has, across a member's scripts.

    Two shapes. An assembled run (afm, indent, pes_tip) declares them up front
    with ``create_box N``. A pre-assembled one (pes_sheet) has no ``create_box``
    at all -- the cell and the type count come from the data file's own
    ``N atom types`` header.

    Taken over the whole member, not per script: a slide script declares nothing
    itself. It reads back ``data/load_$(v_find)N.data``, a file the *run* writes
    and the build does not, under a name only LAMMPS can expand -- so the type
    count it inherits is the one ``system.in`` created.
    """
    for text in texts:
        box = _CREATE_BOX_RE.search(text)
        if box:
            return int(box.group(1))
    counts = [n for text in texts for n, _ in _data_headers(root, text)]
    assert counts, "no create_box and no readable data-file header"
    return max(counts)


def _data_headers(root: Path, text: str) -> List[Tuple[int, Dict[str, float]]]:
    """``(atom types, {xlo, xhi, ylo, yhi})`` for each ``read_data`` target.

    The type count includes any ``extra/atom/types N`` the ``read_data`` line
    reserves. The hetero slide script relies on it: it reads a 24-type data file
    with ``extra/atom/types 1`` and then ``create_atoms 25`` for the virtual
    driving atom, so the system really does have 25 types and
    ``pair_coeff * 25`` is correct.
    """
    out = []
    for ref, tail in _READ_DATA_RE.findall(text):
        target = _resolve(root, ref)
        if target is None or not target.is_file():
            continue
        head = target.read_text(encoding='utf-8')[:4000]
        types = _ATOM_TYPES_RE.search(head)
        bounds = {f"{axis}{end}": float(lo if end == 'lo' else hi)
                  for lo, hi, axis in _DATA_BOUNDS_RE.findall(head)
                  for end in ('lo', 'hi')}
        if types:
            extra = _EXTRA_TYPES_RE.search(tail)
            out.append((int(types.group(1)) + (int(extra.group(1)) if extra
                                               else 0), bounds))
    return out


def _type_tokens(field: str) -> List[int]:
    """Numeric atom types named by one ``pair_coeff`` type field.

    Accepts ``7``, ``1*3`` and ``*``. A bare ``*`` and an open-ended range name
    every type, so they can never be out of bounds and yield nothing to check.
    """
    if '*' not in field:
        return [int(field)] if field.isdigit() else []
    lo, _, hi = field.partition('*')
    return [int(x) for x in (lo, hi) if x.isdigit()]


def _groups_defined(text: str) -> Set[str]:
    """Every group the script has created by this point.

    ``group <name> ...`` is the obvious source; ``read_data ... group <name>``
    is the other one -- LAMMPS creates a group of the atoms it just read, and
    the AFM system script relies on it (``read_data ... group 2D`` is what
    ``velocity 2D create`` later addresses).
    """
    return ({'all'} | set(_GROUP_DEF_RE.findall(text))
            | set(_READ_DATA_GROUP_RE.findall(text)))


def _groups_used(text: str) -> List[Tuple[str, str]]:
    """``(command, group-ID)`` for every group consumer in the script."""
    used = []
    for line in text.split('\n'):
        parts = line.split('#', 1)[0].split()
        index = _GROUP_ARG.get(parts[0]) if parts else None
        if index is not None and len(parts) > index:
            used.append((parts[0], parts[index]))
    return used


def _potential_refs(text: str) -> List[str]:
    """Every potential file named by a many-body ``pair_coeff``.

    A ``sw``/``rebomos``/``airebo`` line reads
    ``pair_coeff * * <style> [n] <file> <elements...>``; ``lj/cut`` takes
    numbers, not a file, so it contributes nothing.
    """
    refs = []
    for _lo, _hi, rest in _PAIR_COEFF_RE.findall(text):
        for token in rest.split():
            if '/' in token and '.' in Path(token).name and token != 'lj/cut':
                refs.append(token)
    return refs


def _members(result) -> List[Path]:
    assert result.ok, result.render()
    return [Path(p) for p in result.created]


def _scripts_of(result) -> List[Tuple[Path, Path]]:
    """``(simulation_root, script)`` for every script of the FIRST member."""
    root = Path(result.simulation_root)
    scripts = sorted(_members(result)[0].rglob('lammps/*.in'))
    assert scripts, _members(result)[0]
    return [(root, s) for s in scripts]


def _builds_of(result) -> List[Tuple[Path, Path, List[Tuple[Path, str]]]]:
    """``(simulation_root, build dir, [(script, full text), ...])`` per build.

    A member is not the unit: ``[2D] layers = [1,2,3]`` gives one member three
    ``L{n}`` builds, and each has its own atom-type count (9, 11, 12 for
    ``afm_config.ini``) because each stacks a different number of sheets.
    Pooling them makes a perfectly valid ``pair_coeff 10 12`` in L3 look
    out of range against L1's ``create_box 9``.
    """
    pairs = _scripts_of(result)
    root = pairs[0][0]
    builds: Dict[Path, List[Tuple[Path, str]]] = {}
    for _r, script in pairs:
        builds.setdefault(script.parent.parent, []).append(
            (script, _script_text(root, script)))
    return [(root, build, texts) for build, texts in sorted(builds.items())]


# --------------------------------------------------------------------------
# The tests
# --------------------------------------------------------------------------

@pytest.mark.parametrize('name,model,members', CASES, ids=[c[0] for c in CASES])
def test_every_example_config_builds_the_number_of_members_it_declares(
        name, model, members, build_example):
    """`generate_package` returns ok=False rather than raising when every member
    fails validation, which is how `indent_campaign.ini` sat 156-for-156 broken
    with nothing noticing."""
    result = build_example(name, model)
    assert result.ok, result.render()
    assert len(result.created) == members, [str(p) for p in result.created]


@pytest.mark.parametrize('name,model,members', CAMPAIGNS,
                         ids=[c[0] for c in CAMPAIGNS])
def test_every_campaign_expands_to_its_full_material_list(
        name, model, members):
    """Counted, not built (decision D4): 156 real members cost 98 s to build and
    render the same script shape. `count_expansion` reads the real file."""
    summary = count_expansion(read_config(EXAMPLES / name), model,
                              base_dir=REPO_ROOT)
    assert summary.configs == members, summary.render()


@pytest.mark.parametrize('name,model,_members', ALL_BUILT,
                         ids=[c[0] for c in ALL_BUILT])
def test_every_pair_coeff_names_an_atom_type_the_system_has(
        name, model, _members, build_example):
    for root, build, texts in _builds_of(build_example(name, model)):
        n_types = _atom_type_count(root, [x for _s, x in texts])
        for script, text in texts:
            for lo_field, hi_field, _rest in _PAIR_COEFF_RE.findall(text):
                for token in _type_tokens(lo_field) + _type_tokens(hi_field):
                    assert 1 <= token <= n_types, (
                        f"{build.name}/{script.name}: pair_coeff names atom "
                        f"type {token}, but the system has {n_types}")


@pytest.mark.parametrize('name,model,_members', ALL_BUILT,
                         ids=[c[0] for c in ALL_BUILT])
def test_every_group_a_fix_names_is_defined(name, model, _members,
                                            build_example):
    for _root, build, texts in _builds_of(build_example(name, model)):
        for script, text in texts:
            defined = _groups_defined(text)
            for command, group in _groups_used(text):
                assert group in defined, (
                    f"{build.name}/{script.name}: {command} uses group "
                    f"'{group}', which nothing defines")


@pytest.mark.parametrize('name,model,_members', ALL_BUILT,
                         ids=[c[0] for c in ALL_BUILT])
def test_the_box_matches_the_requested_sheet_footprint(name, model, _members,
                                                       build_example):
    """Within one lattice constant. The builder replicates whole unit cells and
    picks the count NEAREST the request, so a realised span sits on either side
    of it -- 98.71 for a requested 100 (h-MoS2), 59.13 for 60. Asserting only
    "rounds up" would fail every example."""
    result = build_example(name, model)
    raw = read_config(EXAMPLES / name)
    requested_x = _first_float(raw.get('2D', {}).get('x'))
    requested_y = _first_float(raw.get('2D', {}).get('y'))
    if requested_x is None or requested_y is None:
        pytest.skip(f"{name} does not request a [2D] footprint")

    for root, _build, texts in _builds_of(result):
      for script, text in texts:
        span_x, span_y = _box_span(root, text)
        if span_x is None:
            continue
        for axis, span, requested in (('x', span_x, requested_x),
                                      ('y', span_y, requested_y)):
            assert abs(span - requested) <= _LATTICE_TOL, (
                f"{script.name}: {axis} box span {span:.3f} is more than one "
                f"unit cell away from the requested {requested}")


@pytest.mark.parametrize('name,model,_members', ALL_BUILT,
                         ids=[c[0] for c in ALL_BUILT])
def test_every_potential_file_a_pair_coeff_names_resolves(name, model, _members,
                                                          build_example):
    """From the simulation root, not the script's directory -- the same rule
    `test_orchestration.py` applies to read_data. A path that resolves only
    relative to the script would make every generated job fail at step 0."""
    for root, script in _scripts_of(build_example(name, model)):
        refs = _potential_refs(_script_text(root, script))
        assert refs, f"{script.name}: no many-body potential referenced"
        for ref in refs:
            assert (root / ref).is_file(), (
                f"{script.name}: pair_coeff names '{ref}', which does not "
                f"exist under the simulation root {root}")
            assert not (script.parent / ref).is_file(), (
                f"{script.name}: '{ref}' resolves from the script directory; "
                f"paths must be simulation-root-relative")


@pytest.mark.skipif(not LMP, reason="lmp_serial not on PATH")
def test_the_hetero_example_builds_and_renders(build_example):
    """A hetero build runs lmp_serial as a subprocess (hetero_stack.run_lammps),
    so this one case cannot live in the LAMMPS-free fast tier."""
    name, model, members = HETERO
    result = build_example(name, model)
    assert result.ok, result.render()
    assert len(result.created) == members

    for root, build, texts in _builds_of(result):
        n_types = _atom_type_count(root, [x for _s, x in texts])
        for script, text in texts:
            defined = _groups_defined(text)
            for lo_field, hi_field, _rest in _PAIR_COEFF_RE.findall(text):
                for token in _type_tokens(lo_field) + _type_tokens(hi_field):
                    assert 1 <= token <= n_types, (build.name, script.name,
                                                   token, n_types)
            for command, group in _groups_used(text):
                assert group in defined, (build.name, script.name, command,
                                          group)
            for ref in _potential_refs(text):
                assert (root / ref).is_file(), (build.name, script.name, ref)


# --------------------------------------------------------------------------
# Small helpers used by the box test
# --------------------------------------------------------------------------

#: One lattice constant's slack, either way. The largest cell in
#: frictionsim2d/data/materials is under 7 A, and the builder replicates whole cells to the
#: count nearest the request, so a realised span sits within one cell of it.
_LATTICE_TOL = 7.0


def _first_float(value) -> Optional[float]:
    if isinstance(value, list):
        value = value[0] if value else None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _box_span(root: Path, text: str) -> Tuple[Optional[float], Optional[float]]:
    """The realised in-plane box span, from ``region ... block`` or the data file."""
    region = _REGION_BOX_RE.search(text)
    if region:
        xlo, xhi, ylo, yhi = (float(g) for g in region.groups())
        return xhi - xlo, yhi - ylo
    for _types, bounds in _data_headers(root, text):
        if {'xlo', 'xhi', 'ylo', 'yhi'} <= set(bounds):
            return bounds['xhi'] - bounds['xlo'], bounds['yhi'] - bounds['ylo']
    return None, None
