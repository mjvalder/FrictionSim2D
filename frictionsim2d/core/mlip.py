"""MLIP (machine-learned interatomic potential) descriptor registry.

The single source of truth for every MLIP-specific string emitted into a LAMMPS
deck; ``PotentialManager`` and the builders delegate here rather than branching
on ``pot_type``.

Neither supported framework accepts a partitioned (``NULL``-padded) type map,
so both are scoped to full-cell coverage:

* **DeePMD** carries the model path in the ``pair_style`` line and rejects NULL
  entries (``pair_deepmd.cpp:816``).
* **MACE** carries the model path in the ``pair_coeff`` line and rejects NULL
  type maps (segfault; ACEsuit issue #1314).

Evidence and the capability matrix are in ``documentation/mlip.md``.
"""

from dataclasses import dataclass
from typing import Dict, Sequence

__all__ = [
    'LauncherSpec',
    'MLIPSpec',
    'MLIP_REGISTRY',
    'is_mlip',
    'get_mlip',
    'd3_style_token',
    'd3_coeff_line',
    'model_elements',
    'validate_mlip_cell',
    'check_d3_fits_box',
    'mlip_by_style',
]


@dataclass(frozen=True)
class LauncherSpec:
    """How to launch LAMMPS for a given MLIP framework.

    MACE (ACEsuit fork + Kokkos/libtorch) and DeePMD (pinned mainline +
    deepmd-kit plugin) are two *incompatible* LAMMPS builds, so a job must
    resolve the right binary. ``binary_key`` is the lookup key into an HPC
    profile's ``frameworks:`` overlay -- this module never hard-codes paths.
    """

    binary_key: str
    mpi_ranks: int = 1
    lmp_flags: str = ''
    env_lines: tuple = ()


@dataclass(frozen=True)
class MLIPSpec:
    """Everything the generator needs to know about one MLIP framework."""

    name: str
    model_in_style_line: bool
    file_extensions: tuple
    virtual_atom_element: str
    launcher: LauncherSpec
    # LAMMPS pair_style token, when it differs from the user-facing pot_type:
    # MACE is driven through wcwitt/symmetrix, whose style is `symmetrix/mace`.
    style_name: str = ''
    # Deck lines this style requires, emitted right after `atom_style`; they
    # must precede the box/read_data.
    deck_directives: tuple = ()
    # Does the bare foundation model bind the layers WITHOUT D3? Measured:
    # deepmd (DPA-3.1-3M/Alex2D) has no minimum at all; mace (MACE-MPA-0) binds
    # at 6.717 A, 0.068 J/m^2. Decides whether `dispersion = none` is a
    # runnable control or a broken run (SheetConfig's unbound-stack validator).
    binds_interlayer: bool = False

    @property
    def style_token_name(self) -> str:
        """The LAMMPS pair_style token (``style_name`` if set, else ``name``)."""
        return self.style_name or self.name

    def style_token(self, model_path: str) -> str:
        """Return this framework's contribution to a ``pair_style`` line.

        DeePMD embeds the model path here; MACE's goes into ``pair_coeff``.
        """
        if self.model_in_style_line:
            return f"{self.style_token_name} {model_path}"
        return self.style_token_name

    def coeff_line(self, index_str: str, model_path: str,
                   atom_list: Sequence[str]) -> str:
        """Return a full ``pair_coeff`` line for this framework.

        Args:
            index_str: hybrid sub-style instance suffix, **already carrying its
                leading space** (e.g. ``" 2"``), or ``""`` for a single
                instance. Ignored by MACE, which is full-cell only and
                therefore never multiply instantiated.
            model_path: staged path to the model file.
            atom_list: one entry per LAMMPS atom type -- an element symbol, or
                ``NULL`` for types this instance must ignore (DeePMD only).
        """
        atoms = ' '.join(atom_list)
        if self.model_in_style_line:
            return f"pair_coeff * * {self.style_token_name}{index_str} {atoms}"
        return f"pair_coeff * * {self.style_token_name} {model_path} {atoms}"


MLIP_REGISTRY: Dict[str, MLIPSpec] = {
    'deepmd': MLIPSpec(
        name='deepmd',
        model_in_style_line=True,
        file_extensions=('.pb', '.pth'),
        virtual_atom_element='Mo',
        launcher=LauncherSpec(
            binary_key='deepmd',
            mpi_ranks=1,
            lmp_flags='',
            # The plugin .so lives in a different directory per install method;
            # there is no `deepmd.lmp` module in deepmd-kit v3.
            env_lines=(
                '# deepmd-kit LAMMPS plugin: conda and pip lay it out differently',
                'DEEPMD_PLUGIN_DIR="${CONDA_PREFIX:-/nonexistent}/lib/deepmd_lmp"',
                '[ -d "$DEEPMD_PLUGIN_DIR" ] || DEEPMD_PLUGIN_DIR="$(python -c '
                '\'import deepmd,os;print(os.path.dirname(os.path.dirname('
                'deepmd.__file__)))\')/deepmd/lib"',
                'export LAMMPS_PLUGIN_PATH="$DEEPMD_PLUGIN_DIR"',
            ),
        ),
    ),
    'mace': MLIPSpec(
        name='mace',
        model_in_style_line=False,
        # symmetrix consumes a .json extracted with `symmetrix_extract_mace`.
        file_extensions=('.json',),
        virtual_atom_element='Mo',
        launcher=LauncherSpec(
            binary_key='mace',
            mpi_ranks=1,
            # KOKKOS host (OpenMP) backend, chosen for safety, not speed: the CUDA
            # backend is silently wrong on a stock symmetrix checkout (`mace_types`
            # overflow, see documentation/mlip.md) and the host one is correct.
            # `neigh half` is mandatory: symmetrix/mace/kk requires `newton pair
            # on`, which KOKKOS rejects with `neigh full`. Change the two together.
            lmp_flags='-k on t ${OMP_NUM_THREADS:-8} -sf kk -pk kokkos neigh half',
            env_lines=(
                '# symmetrix/mace is OpenMP-threaded, NOT one-thread-per-rank '
                'like DeePMD -- give it the whole node.',
                'export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${NCPUS:-8}}"',
                'export OMP_PROC_BIND=spread',
                'export OMP_PLACES=threads',
            ),
        ),
        style_name='symmetrix/mace',
        deck_directives=(
            # pair_symmetrix_mace.cpp:172: hard error without a global map.
            'atom_modify     map yes',
            # pair_symmetrix_mace_kokkos.cpp:195 demands `newton pair on`; pairs
            # with `-pk kokkos neigh half` in launcher.lmp_flags.
            'newton          on',
        ),
        binds_interlayer=True,
    ),
}


def is_mlip(pot_type: str) -> bool:
    """Return True if ``pot_type`` names a supported MLIP framework."""
    return (pot_type or '').lower() in MLIP_REGISTRY


def get_mlip(pot_type: str) -> MLIPSpec:
    """Look up the :class:`MLIPSpec` for ``pot_type``.

    Raises:
        ValueError: if ``pot_type`` is not a supported MLIP framework. Callers
            should gate on :func:`is_mlip` first; the error names the valid
            keys so a typo is self-diagnosing.
    """
    key = (pot_type or '').lower()
    try:
        return MLIP_REGISTRY[key]
    except KeyError:
        valid = ', '.join(sorted(MLIP_REGISTRY))
        raise ValueError(
            f"Unknown MLIP pot_type {pot_type!r}. Valid MLIP frameworks are: "
            f"{valid}. Classical potentials (sw, tersoff, reaxff, ...) are not "
            f"handled by the MLIP registry -- check with is_mlip() first."
        ) from None


# ---------------------------------------------------------------------------
# D3 dispersion
# ---------------------------------------------------------------------------

def d3_style_token(settings) -> str:
    """Return the ``dispersion/d3`` ``pair_style`` token.

    The ONE place the D3 argument order is written:
    ``<damping> <functional> <rthr> <cn_thr>``. ``d3_damping`` is emitted
    verbatim and is one of ``original | bj | zerom | bjm`` (``original`` is
    zero-damping). The ACEsuit MACE fork ships an older ``dispersion/d3`` whose
    damping tokens differ, so change the syntax here and only here.

    Args:
        settings: a ``GlobalSettings`` whose ``potential`` block carries
            ``d3_damping``, ``d3_functional``, ``d3_cutoff``, ``d3_cn_cutoff``.
    """
    pot = settings.potential
    return (
        f"dispersion/d3 {pot.d3_damping} {pot.d3_functional} "
        f"{pot.d3_cutoff} {pot.d3_cn_cutoff}"
    )


def d3_coeff_line(elements_per_type: Sequence[str]) -> str:
    """Return the ``dispersion/d3`` ``pair_coeff`` line.

    ``dispersion/d3`` needs a real element for every atom type and rejects NULL,
    so D3 is full-cell only. The caller must already have substituted any
    virtual/drive type with ``MLIPSpec.virtual_atom_element``.
    """
    return f"pair_coeff * * dispersion/d3 {' '.join(elements_per_type)}"


# ---------------------------------------------------------------------------
# Cell-level capability rules
# ---------------------------------------------------------------------------

#: constraint modes that convert the sheet to a molecular atom style with
#: harmonic bonds and ``special_bonds`` exclusions; their interaction with MLIP
#: neighbour lists is unvalidated, so they are refused.
_UNSUPPORTED_CONSTRAINT_MODES = ('atom_bonds', 'com_spring')

#: Pointer put into rejection messages so a user lands on the evidence.
_MLIP_DOC = 'documentation/mlip.md'


def _cell_entries(sheet, components):
    """Normalise ``(label, pot_type, pot_path)`` for every component in a cell."""
    entries = [('sheet', (getattr(sheet, 'pot_type', None) or ''),
                str(getattr(sheet, 'pot_path', '') or ''))]
    entries += [(label, (getattr(cfg, 'pot_type', None) or ''),
                 str(getattr(cfg, 'pot_path', '') or ''))
                for label, cfg in components if cfg is not None]
    return entries


#: Bytes of a model file to read when looking for its element list: the real
#: symmetrix json is ~119 MB but puts `atomic_numbers` in its first few hundred.
_MODEL_HEAD_BYTES = 65536


def model_elements(pot_type: str, model_path) -> "set | None":
    """Element symbols a model covers, or ``None`` when that cannot be known.

    A type map naming an element the model lacks does not error (LAMMPS maps the
    types to the model's first N elements), so coverage is worth checking.

    * **MACE**: the symmetrix ``.json`` carries ``atomic_numbers`` near the top.
    * **DeePMD**: the ``type_map`` is inside a serialized ``.pth`` and reading it
      needs torch, so ``None`` is returned and coverage is left to the run.

    Returns ``None`` rather than raising for anything unreadable, so it can only
    add a diagnosis, never break a working build.
    """
    # Deferred: core must stay ASE-free at import (tests/repo/test_import_weight.py);
    # json/re are only needed on this rarely-hit path.
    import json  # pylint: disable=import-outside-toplevel
    import re  # pylint: disable=import-outside-toplevel

    from ase.data import chemical_symbols  # pylint: disable=import-outside-toplevel

    if pot_type.lower() != 'mace':
        return None
    try:
        with open(model_path, 'r', encoding='utf-8', errors='strict') as fh:
            head = fh.read(_MODEL_HEAD_BYTES)
    except (OSError, UnicodeDecodeError):
        return None

    match = re.search(r'"atomic_numbers"\s*:\s*\[([^\]]*)\]', head)
    if not match:
        return None
    try:
        numbers = json.loads(f"[{match.group(1)}]")
    except json.JSONDecodeError:
        return None
    try:
        return {chemical_symbols[int(z)] for z in numbers}
    except (IndexError, TypeError, ValueError):
        return None


def _labelled_configs(sheet, components):
    """(label, config) for the sheet and every non-None component."""
    yield 'sheet', sheet
    for label, cfg in components or ():
        if cfg is not None:
            yield label, cfg


def validate_mlip_cell(sheet, components=(), constraint_mode='none') -> None:
    """Check the MLIP rules that need to see the *whole* cell.

    ``SheetConfig``'s validator judges the sheet alone; the capability matrix is
    about coverage topology, which needs the tip/substrate/flake alongside it.
    Full-cell MLIP is the only supported topology: one instance spans every
    component. Refused are a cell mixing an MLIP with a classical potential and a
    cell with two MLIP frameworks (``PotentialManager`` refuses them again at
    emission).

    Args:
        sheet: the sheet config (duck-typed: ``pot_type``, ``pot_path``,
            ``dispersion``).
        components: iterable of ``(label, config)`` for every *other* component
            in the cell (tip, sub, flake, second hetero sheet, ...). ``None``
            configs are skipped.
        constraint_mode: ``settings.simulation.constraint_mode``.

    Raises:
        ValueError: naming the rule that rejected the cell.
    """
    entries = _cell_entries(sheet, components)
    sheet_pot = entries[0][1]

    # --- Rule 1: one model file per framework per cell: a pair_style token
    # carries one model path, so a second model could not be expressed.
    models_per_framework: Dict[str, Dict[str, str]] = {}
    for label, pot, path in entries:
        if not is_mlip(pot) or not path:
            continue
        models_per_framework.setdefault(pot.lower(), {}).setdefault(path, label)
    for framework, paths in models_per_framework.items():
        if len(paths) > 1:
            listed = '; '.join(f"{label}: {path}" for path, label in paths.items())
            raise ValueError(
                f"Two different {framework} model files in one cell ({listed}). "
                f"A hybrid pair_style carries one model path per framework, so "
                f"every {framework} instance in a cell must use the SAME model; "
                f"the generator cannot express a second one."
            )

    mlip_labels = [label for label, pot, _ in entries if is_mlip(pot)]
    if not mlip_labels:
        return

    # --- Rule 2: no MLIP may share a cell with a classical component --------
    classical = [(label, pot or '<unset>')
                 for label, pot, _ in entries if not is_mlip(pot)]
    if classical:
        listed = ', '.join(f"{label} ({pot})" for label, pot in classical)
        frameworks = ', '.join(sorted({pot.lower() for _, pot, _ in entries
                                       if is_mlip(pot)}))
        raise ValueError(
            f"Partial-cell MLIP is not available: this cell mixes "
            f"{frameworks} ({', '.join(mlip_labels)}) with classical "
            f"component(s) {listed}. Sharing a cell requires a NULL-padded "
            f"pair_coeff type map, which BOTH frameworks reject — pair_deepmd "
            f"fails with 'Pair hybrid sub-style deepmd is not used' (and "
            f"'pair_coeff 1*2 1*2 deepmd ...' with 'deepmd requires that the "
            f"scale should be set to all atom types, i.e. pair_coeff * *', "
            f"pair_deepmd.cpp:816), and pair_mace segfaults (ACEsuit #1314). "
            f"See {_MLIP_DOC}: make the whole cell one MLIP, or the whole cell "
            f"classical."
        )

    # --- Rule 3a: one MLIP framework per cell: each would need its own
    # `pair_coeff * *`, and the second would silently override the first.
    frameworks = sorted({pot.lower() for _, pot, _ in entries if is_mlip(pot)})
    if len(frameworks) > 1:
        listed = ', '.join(
            f"{label} ({pot.lower()})" for label, pot, _ in entries if is_mlip(pot)
        )
        raise ValueError(
            f"Two MLIP frameworks in one cell ({', '.join(frameworks)}): {listed}. "
            f"Full-cell coverage means a SINGLE instance whose pair_coeff type "
            f"map spans every atom type, and 'pair_coeff * *' is the only form "
            f"either framework accepts — so a second framework's line would "
            f"silently override the first rather than share the cell. Every "
            f"component in a cell must use the SAME MLIP. See {_MLIP_DOC}."
        )

    # --- Rule 3c: `amorph='a'` runs a melt-quench MD at generation time, which
    # can use neither the force-free `pair_style zero` surrogate nor the MLIP.
    amorphous = [label for label, cfg in _labelled_configs(sheet, components)
                 if getattr(cfg, 'amorph', 'c') == 'a']
    if amorphous:
        raise ValueError(
            f"amorph = 'a' is not available under an MLIP "
            f"({', '.join(amorphous)}). Amorphisation is a melt-quench MD run "
            f"performed at GENERATION time, so it needs a potential the "
            f"generating machine can evaluate -- it can use neither the "
            f"`pair_style zero` build surrogate (which exerts no force) nor the "
            f"MLIP (which would require an MLIP-enabled LAMMPS in every "
            f"generation environment and in CI). Use amorph = 'c', or build the "
            f"amorphous structure separately with a classical potential. "
            f"See {_MLIP_DOC}."
        )

    # --- Rule 4: constraint modes whose MLIP neighbour lists are unvalidated
    if is_mlip(sheet_pot) and constraint_mode in _UNSUPPORTED_CONSTRAINT_MODES:
        raise ValueError(
            f"constraint_mode = '{constraint_mode}' is unsupported with an MLIP "
            f"sheet (pot_type = '{sheet_pot}'): it converts the sheet to a "
            f"molecular atom style with harmonic bonds and special_bonds "
            f"exclusions, whose interaction with MLIP neighbour lists is "
            f"unvalidated. Use constraint_mode = 'none'."
        )


#: LAMMPS default neighbour-list skin (``neighbor <skin> bin``) in Angstrom.
_DEFAULT_NEIGHBOR_SKIN = 0.3


def check_d3_fits_box(box_x: float, box_y: float, d3_cutoff: float,
                      skin: float = _DEFAULT_NEIGHBOR_SKIN) -> None:
    """Reject a D3 overlay whose cutoff does not fit the periodic box.

    ``pair_style dispersion/d3`` sets the communication cutoff to
    ``d3_cutoff + skin``, and LAMMPS cannot communicate ghosts further than one
    box length, so a wider cutoff aborts the run deep inside LAMMPS with
    ``Communication cutoff ... cannot exceed periodic box length``. Small
    screening cells are where this bites (``x = y = 15`` builds 15.9 x 16.5 A
    against the default 16.0).

    Only the periodic in-plane dimensions are checked; z carries vacuum and is
    not periodic in the sliding geometries.

    Raises:
        ValueError: naming the offending cutoff and both ways out.
    """
    required = d3_cutoff + skin
    shortest = min(box_x, box_y)
    if shortest >= required:
        return
    raise ValueError(
        f"d3_cutoff = {d3_cutoff} A needs a periodic box of at least "
        f"{required:.1f} A (cutoff + {skin} A neighbour skin) in x and y, but "
        f"this cell is {box_x:.1f} x {box_y:.1f} A -- the shortest side is "
        f"{shortest:.1f} A. LAMMPS cannot send ghost atoms further than one "
        f"box length, so the run would abort with 'Communication cutoff ... "
        f"cannot exceed periodic box length'. Either reduce "
        f"settings.potential.d3_cutoff below {shortest - skin:.1f}, or "
        f"increase the sheet footprint (x/y) past {required:.1f} A."
    )


def mlip_by_style(style_token: str):
    """Resolve a LAMMPS ``pair_style`` token (``symmetrix/mace``) to its :class:`MLIPSpec`.

    Returns ``None`` if the token is not an MLIP style.
    """
    token = (style_token or '').strip()
    for spec in MLIP_REGISTRY.values():
        if token == spec.style_token_name:
            return spec
    return None
