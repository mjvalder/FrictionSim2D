"""MLIP (machine-learned interatomic potential) descriptor registry.

This module is the single source of truth for every MLIP-specific string that
FrictionSim2D emits into a LAMMPS deck. ``PotentialManager`` and the builders
delegate here rather than branching on ``pot_type`` themselves, so that adding a
framework -- or correcting a syntax detail after a real LAMMPS build disagrees
with the documentation -- is a one-file change.

Two frameworks are supported. **Neither accepts a partitioned (``NULL``-padded)
type map**, so both are scoped to full-cell coverage:

* **DeePMD** carries the model path in the ``pair_style`` line. It *rejects*
  NULL entries -- a NULL-padded map yields ``ERROR: Pair hybrid sub-style
  deepmd is not used``, and restricting by type range yields ``deepmd requires
  that the scale should be set to all atom types, i.e. pair_coeff * *``
  (``pair_deepmd.cpp:816``). Measured on real hardware, 2026-08-10.
* **MACE** carries the model path in the ``pair_coeff`` line and *rejects* NULL
  type maps (segfault; ACEsuit issue #1314).

See ``docs/superpowers/specs/2026-08-11-mlip-post-m0-design.md`` (v4) §1 for the
evidence and §3 for the collapsed capability matrix. v4 supersedes the earlier
``2026-08-06`` design, whose setups 1, 3 and 4 all assumed DeePMD NULL support
and are struck.
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
    # LAMMPS pair_style token, when it differs from the user-facing pot_type.
    # MACE is driven through wcwitt/symmetrix, whose style is `symmetrix/mace`,
    # not `mace` (that is the separate, unbuilt ACEsuit fork).
    supports_null: bool
    supports_overlay: bool
    file_extensions: tuple
    virtual_atom_element: str
    launcher: LauncherSpec
    # LAMMPS pair_style token, when it differs from the user-facing pot_type.
    # MACE is driven through wcwitt/symmetrix, whose style is `symmetrix/mace`,
    # not `mace` (that is the separate, unbuilt ACEsuit fork).
    style_name: str = ''
    # extra keywords appended to the pair_style token (ACEsuit MACE needed
    # `no_domain_decomposition`; symmetrix takes none).
    style_args: str = ''
    # Deck lines this style requires, emitted right after `atom_style`.
    # They must precede the box/read_data, which is why they live here rather
    # than being appended by the caller.
    deck_directives: tuple = ()
    # Does the bare foundation model reproduce interlayer binding WITHOUT D3?
    # Measured here, not assumed -- the two frameworks genuinely differ:
    #   deepmd (DPA-3.1-3M/Alex2D, 2026-08-11): NO minimum anywhere in the scan,
    #     5.7e-7 J/m^2. Layers drift apart; interlayer numbers are meaningless.
    #   mace (MACE-MPA-0-medium, 2026-08-24): a real minimum at 6.717 A with
    #     0.068 J/m^2 -- weak versus the ~0.25-0.30 literature value, but bound.
    # This drives whether `dispersion = none` is a runnable control or a
    # guaranteed-garbage run (see SheetConfig's unbound-stack validator).
    binds_interlayer: bool = False
    notes: str = ''

    @property
    def style_token_name(self) -> str:
        """The LAMMPS pair_style token (``style_name`` if set, else ``name``)."""
        return self.style_name or self.name

    def style_token(self, model_path: str) -> str:
        """Return this framework's contribution to a ``pair_style`` line.

        DeePMD embeds the model path here; MACE does not (its model path goes
        into ``pair_coeff``) and instead needs ``no_domain_decomposition``.
        """
        if self.model_in_style_line:
            return f"{self.style_token_name} {model_path}"
        return f"{self.style_token_name} {self.style_args}".strip()

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
        supports_null=False,
        supports_overlay=True,
        file_extensions=('.pb', '.pth'),
        virtual_atom_element='Mo',
        launcher=LauncherSpec(
            binary_key='deepmd',
            mpi_ranks=1,
            lmp_flags='',
            # The plugin .so lives in a DIFFERENT directory per install method,
            # and both layouts are real (v4 spec §1). Verified 2026-08-10:
            #   conda: $CONDA_PREFIX/lib/deepmd_lmp/dpplugin.so
            #   pip:   <site-packages>/deepmd/lib/libdeepmd_lmpplugin.so
            # There is NO `deepmd.lmp` module in deepmd-kit v3 -- the
            # widely-copied `import deepmd.lmp` incantation raises
            # ModuleNotFoundError, the export then evaluates to a traceback
            # string, and LAMMPS reports "Loaded 0 plugins" followed by
            # "Unrecognized pair style 'deepmd'".
            env_lines=(
                '# deepmd-kit LAMMPS plugin: conda and pip lay it out '
                'differently (v4 spec section 1)',
                'DEEPMD_PLUGIN_DIR="${CONDA_PREFIX:-/nonexistent}/lib/deepmd_lmp"',
                '[ -d "$DEEPMD_PLUGIN_DIR" ] || DEEPMD_PLUGIN_DIR="$(python -c '
                '\'import deepmd,os;print(os.path.dirname(os.path.dirname('
                'deepmd.__file__)))\')/deepmd/lib"',
                'export LAMMPS_PLUGIN_PATH="$DEEPMD_PLUGIN_DIR"',
            ),
        ),
        notes=(
            "TRAP: never emit the multi-model form 'pair_style deepmd m1 m2'. "
            "That is model-deviation (uncertainty) mode, in which only the "
            "first model drives the forces -- silently wrong physics, not an "
            "error. "
            "NULL TYPE MAPS ARE REJECTED (v4 spec §1, measured 2026-08-10): "
            "'pair_coeff * * deepmd Mo S NULL NULL NULL NULL' dies with "
            "'ERROR: Pair hybrid sub-style deepmd is not used' -- for a single "
            "instance as well as three, so NULL is the cause, not "
            "multi-instancing. Restricting by type range instead "
            "('pair_coeff 1*2 1*2 deepmd Mo S') dies with 'deepmd requires "
            "that the scale should be set to all atom types, i.e. "
            "pair_coeff * *' (pair_deepmd.cpp:816). So pair_coeff MUST be "
            "'* *' with a FULL element list, one entry per LAMMPS type, and "
            "partial-cell DeePMD is unavailable. "
            "SILENT-WRONG TRAP: omitting the element list "
            "('pair_coeff * * deepmd') does NOT error -- LAMMPS types are "
            "mapped to the model's first N elements (H, He, Li, ...), giving "
            "-61.89 eV instead of -256.90 eV on the reference 36-atom MoS2 "
            "trilayer. Requires the deepmd-kit LAMMPS plugin: "
            "LAMMPS_PLUGIN_PATH must be exported at run time."
        ),
    ),
    'mace': MLIPSpec(
        name='mace',
        model_in_style_line=False,
        supports_null=False,
        supports_overlay=True,
        # symmetrix consumes a .json extracted from a torch .model via
        #   symmetrix_extract_mace -m <model> --chemical-symbols Mo S
        file_extensions=('.json',),
        virtual_atom_element='Mo',
        launcher=LauncherSpec(
            binary_key='mace',
            mpi_ranks=1,
            # KOKKOS **HOST (OpenMP)** backend -- `-k on t N`, not `-k on g 1`.
            # This default is chosen for SAFETY, not speed. Measured
            # 2026-08-31 on an h-MoS2 bilayer (24 atoms) and trilayer (810):
            #   plain CPU `symmetrix/mace`      -173.40779 / -5850.0183 eV
            #   Kokkos host  `.../kk`           -173.40779 / -5850.0183 eV
            #   Kokkos CUDA  `.../kk` UNPATCHED -126.5489  / -4785.9659 eV
            #   Kokkos CUDA  `.../kk` PATCHED   -173.40779 / -5850.0183 eV
            # The CUDA backend is silently wrong -- and correct once the
            # `mace_types` overflow is patched (root cause in `notes`). But
            # that fix is NOT upstream, so a stock symmetrix checkout is
            # unsafe on CUDA and we must not default to it. The host backend
            # is correct either way, and still ~4x faster than the plain CPU
            # pair style (0.64 vs 2.58 s/step, 810 atoms, 8 threads).
            # To opt into CUDA (~2.3x faster again: 0.108 vs 0.251 s/step at
            # 810 atoms) you must BOTH apply the patch AND verify the
            # reference energy -- see `preflight.sh` in the campaign folder.
            # `neigh half` is mandatory: symmetrix/mace/kk requires
            # `newton pair on`, and KOKKOS rejects newton-on with neigh full
            # ("Must use 'newton off' with KOKKOS package option 'neigh full'",
            # kokkos.cpp:713). Measured 2026-08-24. Change the two together.
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
        style_args='',
        deck_directives=(
            # pair_symmetrix_mace.cpp:172 -- hard error without a global map.
            'atom_modify     map yes',
            # pair_symmetrix_mace_kokkos.cpp:195 demands `newton pair on`,
            # while KOKKOS defaults to `neigh full` which demands newton off.
            # Resolved by pairing this with `-pk kokkos neigh half` in
            # launcher.lmp_flags -- change the two together or runs abort.
            'newton          on',
        ),
        # Unlike DPA-3, MACE-MPA-0 does bind a bilayer unaided: 6.717 A,
        # 0.068 J/m^2 (bench 2026-08-24). Under-binds versus 0.25-0.30
        # literature, and D3 then OVER-binds it (0.474 J/m^2, 1.6-1.9x), so
        # the no-D3 arm is a legitimate control here rather than a broken run.
        binds_interlayer=True,
        notes=(
            "Driven through wcwitt/symmetrix (`pair_style symmetrix/mace`), "
            "NOT the ACEsuit fork's `pair_style mace` -- symmetrix is the "
            "implementation ML-MIX supports for hybrid ML/MM, whereas "
            "`mliappy mace` is listed INCOMPATIBLE with it. Needs LAMMPS "
            ">= 10 Sep 2025. Because that LAMMPS is modern, `dispersion/d3` "
            "is NATIVE -- no backport, unlike DeePMD.\n"
            "TRAPS, measured 2026-08-24 on an h-MoS2 trilayer (810 atoms):\n"
            "  * Requires `atom_modify map yes`, else a hard run-time error "
            "(pair_symmetrix_mace.cpp:172).\n"
            "  * The model path goes in pair_coeff, never in pair_style.\n"
            "  * A NON-hybrid coeff line must DROP the style token: "
            "`pair_coeff * * model.json Mo S` works, whereas "
            "`pair_coeff * * symmetrix/mace model.json Mo S` fails. Under "
            "hybrid/overlay the token is required as usual.\n"
            "  * The conda `mlip` env ships its own liblammps.so (bundled "
            "with deepmd-kit) that shadows this build -- put the build dir "
            "first on LD_LIBRARY_PATH or it aborts with `undefined symbol: "
            "lammps_plugin_finalize`.\n"
            "  * Runs exit with a heap abort ('corrupted size vs. prev_size') "
            "AFTER results are written: benign for the physics, but the "
            "non-zero exit code breaks PBS job dependencies.\n"
            "ROOT CAUSE OF BOTH THE HEAP ABORT AND THE CUDA ERROR "
            "(found 2026-08-31, one bug): symmetrix allocates its type map "
            "as `mace_types = View<int*>(\"mace_types\", "
            "mace->atomic_numbers.size())` -- one slot per MODEL ELEMENT -- "
            "then fills one slot per LAMMPS ATOM TYPE. Those agree only when "
            "ntypes == n_elements, which holds for upstream's examples and "
            "fails for any multi-layer cell: an h-MoS2 trilayer has 7 types "
            "against a 2-element (Mo,S) model, so every pair_coeff writes 5 "
            "ints past a 2-int allocation. Consequences:\n"
            "  * On CUDA only 2 valid entries reach the device, so types 3..7 "
            "read out-of-bounds DEVICE memory and get the wrong element -- "
            "the silent 27% energy error (-126.5489 vs -173.40779 eV). On the "
            "host the mirror aliases the view, so the overflowed values read "
            "back intact and the energy is right despite the corruption. That "
            "asymmetry is why it looked like a CUDA-only kernel bug.\n"
            "  * The corrupted heap kills anything that allocates soon after: "
            "hybrid/overlay + dispersion/d3 died in 4.6 s in "
            "'malloc_consolidate(): invalid chunk size'.\n"
            "The CPU pair style is unaffected (std::vector::push_back). "
            "Fix: size the view by the type-map length (narg-3). NOT upstream "
            "as of origin/main 0d86e1e or origin/develop; the patch ships as "
            "campaign_mace/symmetrix-mace_types-overflow.patch.\n"
            "A SEPARATE, still-unfixed teardown abort remains: the binary "
            "exits non-zero (rc=134) even on a bare `lmp -h`, after the "
            "output is written. Tolerate it by checking the run completed, "
            "never by blanket-ignoring the exit code.\n"
            "NULL type maps remain unavailable (full-cell only), so the "
            "capability matrix is unchanged."
        ),
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

    .. warning::

       **This is the ONE place the D3 argument order is written.** The order
       below is ``<damping> <functional> <rthr> <cn_thr>``, which
       ``PairDispersionD3::settings`` confirms and Milestone-0 item E5 executed
       (``pair_style dispersion/d3 bj pbe 16.0 8.0`` on a 24-atom bilayer,
       ``E_pair = -8.7249526 eV``). Both cutoffs are squared internally and
       ``units metal`` is required; damping is one of
       ``original | zero | zerom | bj | bjm`` (``original`` == ``zero``).
       The ACEsuit MACE fork ships an older ``dispersion/d3`` than pinned
       mainline (pre-rename damping tokens), so the accepted syntax can still
       differ on that build. If it ever has to change, change it **here and
       only here** -- every emitted deck routes through this function.

       Note that ``dispersion/d3`` does not exist before LAMMPS
       ``stable_22Jul2025``, while deepmd-kit is ABI-locked to
       ``2024.08.29``; reaching both in one binary needs the backported build
       of v4 spec §4.

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

    Stock ``pair_style dispersion/d3`` requires a **real element for every atom
    type** -- it does not accept NULL. That is precisely why D3 is restricted
    to full-cell MLIP coverage (a partial-cell overlay would double-count
    dispersion over the SW/LJ pairs). Any virtual/drive type must already have
    been substituted by the caller with ``MLIPSpec.virtual_atom_element``.
    """
    return f"pair_coeff * * dispersion/d3 {' '.join(elements_per_type)}"


# ---------------------------------------------------------------------------
# Cell-level capability rules (v4 spec §3 and §6)
# ---------------------------------------------------------------------------

#: constraint modes that convert the sheet to a molecular atom style with
#: harmonic bonds and ``special_bonds`` exclusions. Their interaction with MLIP
#: neighbour lists is unvalidated, so they are refused rather than silently
#: producing suspect physics.
UNSUPPORTED_MLIP_CONSTRAINT_MODES = ('atom_bonds', 'com_spring')

#: Pointer put into every rejection message so a surprised user lands on the
#: evidence rather than on a rule that looks arbitrary.
V4_SPEC = 'docs/superpowers/specs/2026-08-11-mlip-post-m0-design.md'
V6_SPEC = 'docs/superpowers/specs/2026-09-03-mlip-multicomponent-fullcell-design.md'


def _cell_entries(sheet, components):
    """Normalise ``(label, pot_type, pot_path)`` for every component in a cell."""
    entries = [('sheet', (getattr(sheet, 'pot_type', None) or ''),
                str(getattr(sheet, 'pot_path', '') or ''))]
    entries += [(label, (getattr(cfg, 'pot_type', None) or ''),
                 str(getattr(cfg, 'pot_path', '') or ''))
                for label, cfg in components if cfg is not None]
    return entries


#: How much of a model file to read when looking for its element list. The real
#: symmetrix json is ~119 MB but puts `atomic_numbers` in its first few hundred
#: bytes, so a bounded prefix read keeps this check free.
_MODEL_HEAD_BYTES = 65536


def model_elements(pot_type: str, model_path) -> "set | None":
    """Element symbols a model covers, or ``None`` when that cannot be known.

    Used for the coverage check of v6 §4 rule 3b. Coverage matters more here
    than for a typical validation because the failure it prevents is **silent**:
    v4 E7 showed that a type map naming an element the model does not have does
    not error -- LAMMPS maps the types to the model's first N elements and
    returns a plausible, wrong energy.

    The two frameworks differ in what can be checked:

    * **MACE** -- the symmetrix ``.json`` carries ``atomic_numbers`` in plain
      JSON near the top of the file, so the answer is exact and cheap.
    * **DeePMD** -- the ``type_map`` is inside a serialized ``.pth``; reading it
      needs torch, which generation and CI must not depend on (v4 §3.3). Returns
      ``None``, and coverage is deferred to the run-time gate.

    Returns ``None`` rather than raising for anything unreadable (a stub, a
    binary blob, a missing file), so this can only ever *add* a diagnosis --
    never break a build that would otherwise have worked.
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

    ``SheetConfig``'s own validator can only judge the sheet in isolation. The
    capability matrix, though, is about **coverage topology**: whether every
    real atom type in the cell is covered by ONE MLIP instance with a full type
    map. That needs the tip/substrate/flake alongside the sheet, which is why
    this lives here and is called by the builders rather than by Pydantic.

    After Milestone 0 the matrix is short (v4 spec §3): full-cell MLIP is the
    only supported topology. v4 §6 additionally limited that to a
    **single-component** cell, because the builders instantiated one MLIP per
    component with NULL-padded maps. They now emit ONE instance for the whole
    cell (``TypeRegistry.build_full_map``), so a tip+sheet+substrate cell is
    supported and that limit is gone (v6 §3).

    What remains refused is a cell that is not *uniform*: an MLIP beside a
    classical potential (rule 2), or two MLIP frameworks together (rule 3a).
    Both are caught here rather than at 3 a.m. on the cluster —
    ``PotentialManager`` refuses them again at emission, because by then the
    full type map would otherwise absorb the foreign component's types
    silently.

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

    # --- Rule 1: one model file per framework per cell ----------------------
    # One pair_style token carries one model path per framework, so a second,
    # different model of the same framework simply cannot be expressed —
    # PotentialManager would silently use the first for every instance.
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
            f"See {V4_SPEC} §3 (evidence E2/E3 in §1): make the whole cell one "
            f"MLIP, or the whole cell classical."
        )

    # --- Rule 3a: one MLIP framework per cell -------------------------------
    # v4 §6 refused every multi-component cell, because the builders emitted one
    # instance PER COMPONENT with NULL-padded maps. They now emit ONE instance
    # for the whole cell (`TypeRegistry.build_full_map`), so a uniform
    # multi-component cell is buildable and that refusal is gone (v6 §3/§4).
    #
    # What a single instance cannot do is span two FRAMEWORKS. Each would need
    # its own `pair_coeff * *` — the only form either accepts — and the second
    # simply overrides the first, silently.
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
            f"component in a cell must use the SAME MLIP. See {V6_SPEC} §4."
        )

    # --- Rule 3c: amorphous components cannot be built under an MLIP --------
    # `amorph='a'` triggers a melt-quench MD run at GENERATION time
    # (components.py, via get_single_component_commands). That run's result
    # depends on forces, so it can use neither the `pair_style zero` surrogate
    # (v4 §3.3 forbids it for anything that integrates) nor the MLIP itself,
    # which would require an MLIP-enabled LAMMPS on every generation machine and
    # in CI. Refuse rather than emit a deck that is wrong for both frameworks.
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
            f"See {V6_SPEC} §5."
        )

    # --- Rule 4: constraint modes the MLIP neighbour lists are unvalidated for
    if is_mlip(sheet_pot) and constraint_mode in UNSUPPORTED_MLIP_CONSTRAINT_MODES:
        raise ValueError(
            f"constraint_mode = '{constraint_mode}' is unsupported with an MLIP "
            f"sheet (pot_type = '{sheet_pot}'): it converts the sheet to a "
            f"molecular atom style with harmonic bonds and special_bonds "
            f"exclusions, whose interaction with MLIP neighbour lists is "
            f"unvalidated. Use constraint_mode = 'none'."
        )


#: LAMMPS default neighbour-list skin (``neighbor <skin> bin``) in Angstrom.
DEFAULT_NEIGHBOR_SKIN = 0.3


def check_d3_fits_box(box_x: float, box_y: float, d3_cutoff: float,
                      skin: float = DEFAULT_NEIGHBOR_SKIN) -> None:
    """Reject a D3 overlay whose cutoff does not fit the periodic box.

    ``pair_style dispersion/d3`` sets the communication cutoff to
    ``d3_cutoff + skin``. LAMMPS cannot communicate ghosts further than one box
    length in a single shell, so a cutoff wider than the shortest periodic
    dimension aborts the run -- deep inside LAMMPS, long after generation::

        ERROR: Communication cutoff for comm_style tiled cannot exceed
        periodic box length

    Measured 2026-08-11: an ``x = y = 15`` sheet builds a 15.9 x 16.5 A box and
    fails against the default ``d3_cutoff = 16.0``; ``x = y = 20`` (19.1 x 22.1)
    clears it. Small screening cells are exactly the regime where this bites,
    so it is worth diagnosing up front.

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
    """Resolve a LAMMPS ``pair_style`` token back to its :class:`MLIPSpec`.

    ``is_mlip`` keys off the user-facing ``pot_type`` (``mace``), but emitted
    LAMMPS lines carry the *style* token, which for MACE is ``symmetrix/mace``.
    Returns ``None`` if the token is not an MLIP style.
    """
    token = (style_token or '').strip()
    for spec in MLIP_REGISTRY.values():
        if token == spec.style_token_name:
            return spec
    return None
