# Milestone 0 — MLIP feasibility runbook

Milestone 0 is the **feasibility gate** for MLIP support: mechanisms that are
documented or community-confirmed but **unproven in our stack** get proven or
disproven here, with hand-written LAMMPS decks — no FrictionSim2D code in the
loop, so a failure here is a failure of LAMMPS/the model, never of the builder.

> **This runbook has been run, and it changed the design.** The current
> capability matrix lives in
> [`docs/superpowers/specs/2026-08-11-mlip-post-m0-design.md`](../docs/superpowers/specs/2026-08-11-mlip-post-m0-design.md)
> (**v4**), which supersedes the original `2026-08-06` spec. The headline
> result: `pair_deepmd` rejects NULL type maps, so **partial-cell MLIP is
> impossible for both frameworks** and the original setups 1, 3 and 4 are
> struck — not "conditional". What survives, and is verified on hardware, is
> **full-cell DeePMD, base and + D3**. Read the results below as the evidence
> for v4, not as open questions against the v3 matrix.

Nothing produced by the framework work (Phases 1–5) may be described as
"verified" for a Conditional cell until the corresponding item below passes.

Run everything on the GPU workstation, or in a short `gpu72` job. The systems
are tiny (24–36 atoms), so every item is seconds of GPU time.

## Prerequisites

### 1. The two LAMMPS builds

See [`scripts/hpc/README.md`](../scripts/hpc/README.md). Mutually incompatible
builds are needed:

```bash
cd scripts/hpc
PREFIX=$HOME/opt TARGET=workstation ./build_lammps_deepmd.sh
PREFIX=$HOME/opt TARGET=workstation ./build_lammps_deepmd_d3.sh   # deepmd AND d3
PREFIX=$HOME/opt TARGET=workstation GPU_ARCH=AMPERE86 ./build_lammps_mace.sh
```

Resulting binaries (referenced throughout as `$LMP_DEEPMD` / `$LMP_MACE`):

```bash
export PREFIX=$HOME/opt
export LMP_DEEPMD="$PREFIX/lammps-deepmd/bin/lmp"
export LMP_DEEPMD_D3="$PREFIX/lammps-deepmd-d3/bin/lmp"
export LMP_MACE="$PREFIX/lammps-mace/build/lmp"
```

`$LMP_DEEPMD_D3` is the **backported** build and is the only one that can run
any deck combining `deepmd` with `dispersion/d3` — prefer it for every DeePMD
item. See "RESOLVED: deepmd + dispersion/d3 in one binary" below.

Use `GPU_ARCH=ADA89` on an L40S (gpu72), `AMPERE80` on an A100.

### 2. Environment exports

**DeePMD** — the framework is a *runtime plugin*, so without this export
`pair_style deepmd` is simply an unknown style:

```bash
source "$PREFIX/deepmd-venv/bin/activate"
export LAMMPS_PLUGIN_PATH=$(python -c "import deepmd,os;print(os.path.join(os.path.dirname(deepmd.__file__),'lib'))")
```

**MACE** — no plugin path, but every invocation needs the Kokkos flags
`-k on g 1 -sf kk` (one GPU, one MPI rank; the pair style runs with
`no_domain_decomposition`):

```bash
source "$PREFIX/mace-venv/bin/activate"
"$LMP_MACE" -k on g 1 -sf kk -in <deck>
```

### 3. Geometries

```bash
cd scripts/milestone0
python make_geometries.py --outdir .
```

writes the three data files every deck reads:

| file | atoms | atom types | used by |
|---|---|---|---|
| `bilayer_mos2.data` | 24 | 4 | M0.2, M0.3, M0.3b, M0.4 (reference energy) |
| `trilayer_mos2.data` | 36 | 6 | M0.1, M0.6 |
| `bilayer_plus_virtual.data` | 25 | 5 | M0.4 |

**Atom type layout** (one Mo type and one S type *per layer* — this is what
makes per-layer NULL maps addressable, and ASE would otherwise emit only two
types):

```
1 = Mo layer 1   2 = S layer 1
3 = Mo layer 2   4 = S layer 2
5 = Mo layer 3   6 = S layer 3      (trilayer only)
5 = virtual drive atom              (bilayer_plus_virtual only)
```

`bilayer_plus_virtual.data` is the M0.3 bilayer with an identical cell and
identical first 24 atom positions, plus one extra atom. That is a hard
requirement of M0.4's constant-offset test.

### 4. Models

See [Model acquisition](#model-acquisition) below. The decks expect the model
files in the working directory under these names:

| deck name | produced by |
|---|---|
| `mace.model-lammps.pt` | `mace_create_lammps_model` on `mace-mp-0b3-medium` |
| `frozen_model.pth` | `dp --pt freeze` on the DPA checkpoint |
| `mace-mliap_lammps.pt` | `mace_create_lammps_model --format=mliap` (M0.6 only) |

## Items

Numbering matches spec §7.

### M0.1 — DeePMD multi-instance hybrid with NULL maps

**Objective.** Prove that two, then three, *independent* `deepmd` sub-style
instances can coexist in one `pair_style hybrid` with NULL type maps and
`lj/cut`. This is the partial-cell mechanism behind setups 1/3/4 — the whole
per-layer MLIP route depends on it. It also confirms that indexed sub-style
suffixing (`deepmd 1`, `deepmd 2`, `deepmd 3`) is accepted, and that no
model-deviation machinery is triggered (spec: never emit
`pair_style deepmd m1 m2`).

**Command.**

```bash
source "$PREFIX/deepmd-venv/bin/activate"
export LAMMPS_PLUGIN_PATH=$(python -c "import deepmd,os;print(os.path.join(os.path.dirname(deepmd.__file__),'lib'))")
cd scripts/milestone0
"$LMP_DEEPMD" -in m0_1_deepmd_multi_hybrid.in
```

Run the two-instance variant first by deleting the third `deepmd` from the
`pair_style` line, dropping the `deepmd 3` coeff line, and folding layer 3 into
the second instance's map — if two instances fail, three will not help.

**PASS.** 20 steps complete without error; total energy and all forces finite;
the log lists three independent sub-styles; no model-deviation output. If it
passes, matrix rows 2 and 3 (partial-cell) flip from *Conditional* to **Yes**
for DeePMD base.

### M0.2 — DeePMD full-cell + D3 under `hybrid/overlay`

**Objective.** The genuinely open item: `deepmd` and `dispersion/d3` overlaid on
the same full-cell system. There are community segfault reports for this
combination. This deck is also where the **`dispersion/d3` argument order is
pinned** for the pinned-mainline build — the value the registry (`d3_style_token`
in `src/core/mlip.py`) will emit.

**Command.** Requires the backported binary — no other build has both styles.

```bash
export LAMMPS_PLUGIN_PATH=/home/mv923/miniconda3/envs/mlip/lib/deepmd_lmp
export LD_LIBRARY_PATH=/home/mv923/miniconda3/envs/mlip/lib
env -u PYTHONPATH "$LMP_DEEPMD_D3" -in m0_2_deepmd_overlay_d3.in
```

The deck's `pair_style` line ships as

```
pair_style hybrid/overlay deepmd dpa3_alex2d.pth dispersion/d3 bj pbe 16.0 8.0
```

**PASS.** 20 steps complete, no segfault, finite energy/forces, and the D3
contribution is non-zero — read it directly from `compute pair dispersion/d3`
rather than differencing two runs. This flips matrix row 1 ("Full-cell MLIP")
column "DeePMD + D3" to **Yes**.

**RESULT: PASS**, 2026-08-11. `E_pair = -180.15327 eV` (`c_edp = -171.42832`,
`c_ed3 = -8.7249526`) against `-171.42832 eV` for the same cell with no
overlay. No segfault; TotEng drifts 9e-5 eV over 20 NVE steps. The community
segfault reports for this combination did not reproduce.

### M0.3 / M0.3b — MACE full-cell base, and MACE + D3

**Objective.** M0.3 establishes the MACE full-cell baseline (`pair_style mace`,
no hybrid) and produces the reference `E_pair` that M0.4 subtracts. M0.3b tests
the community-confirmed (matsci #62930) `hybrid/overlay mace + dispersion/d3`
pattern and pins the D3 argument order **for the ACEsuit fork**, which ships an
older `dispersion/d3` than pinned mainline and may use pre-rename damping
tokens. The two builds are allowed to disagree — that is exactly why spec §1.2
makes `d3_damping` a *canonical* name the registry translates per build.

**Commands.**

```bash
source "$PREFIX/mace-venv/bin/activate"
cd scripts/milestone0
"$LMP_MACE" -k on g 1 -sf kk -in m0_3_mace_fullcell.in
"$LMP_MACE" -k on g 1 -sf kk -in m0_3b_mace_overlay_d3.in
```

**PASS.** Both complete 20 steps with finite energy/forces; M0.3b's `E_pair`
differs from M0.3's by a non-zero dispersion energy. Record M0.3's step-0
`E_pair` in the notes column — M0.4 needs it. M0.3 passing keeps row-1 "MACE
base" at **Yes**; M0.3b passing flips row-1 "MACE + D3" to **Yes**.

### M0.4 — Virtual-atom compatibility (spec §1.4)

**Objective.** The go/no-go for keeping `drive_method = virtual_atom` — the
project's preferred drive — with MLIPs and D3. The virtual atom has no chemical
element, which `pair_style mace` cannot NULL-map and `dispersion/d3` cannot
accept. The proposed mechanism is (a) map it to a **placeholder element** that
exists in the model, and (b) `neigh_modify exclude` every pair involving it, so
the MLIP graph and the D3 sum never see it. An atom with an empty neighbour list
should contribute only its constant isolated-atom energy. Graph models generally
handle isolated nodes — but this is precisely the class of claim that must be
measured, not assumed.

**Commands.**

```bash
"$LMP_MACE" -k on g 1 -sf kk -in m0_4_virtual_exclude.in
```

Then repeat the same deck twice more, editing in place:

- **DeePMD variant** — replace the pair block with
  `pair_style deepmd frozen_model.pth` and
  `pair_coeff * * Mo S Mo S Mo`, and run with `$LMP_DEEPMD` (plugin path
  exported).
- **D3 variant** — take M0.3b's `hybrid/overlay` pair block and append ` Mo`
  to both the `mace` and the `dispersion/d3` coeff lines so type 5 is covered.

**PASS** (all three variants):

1. Potential energy differs from the corresponding no-virtual run
   (M0.3 / M0.2-without-overlay) by a **constant** — the isolated-atom energy of
   the placeholder element. Constant means: identical at step 0 and after
   displacing the virtual atom, not merely "close".
2. The force on the type-5 atom is **exactly zero** in all three components.
   The deck's `thermo_style` prints `c_fvirt[1..3]` (`compute reduce maxabs`)
   for this.
3. 20 steps of dynamics complete with the virtual atom present.

**FAIL consequence.** Validation falls back to *requiring* `smd` or `fix_move`
for MLIP/D3 runs: `virtual_atom` becomes a rejected `drive_method` in those
combinations, and the §1.4 placeholder/exclusion emission in
`PotentialManager.write_file` is dropped.

**RESULT (DeePMD variant): PASS**, 2026-08-11, on `$LMP_DEEPMD_D3`. Offset
**-4.64264 eV**, exactly constant across two displacements of the virtual
atom; max|F| on type 5 = 0,0,0; 20 NVE steps stable. Together with the
already-passing D3 variant this keeps `virtual_atom` available for full-cell
MLIP runs — see "VERIFIED: the virtual atom survives an MLIP" below for the
one consequence (absolute energies carry a constant placeholder offset). The
MACE variant remains unrun.

### M0.5 — `smd` and `fix_move` drive paths

**Objective.** The fallback drive paths must work regardless of M0.4's outcome,
and they are currently under-exercised. This item is **framework-level, not a
hand-written deck**: it is golden + smoke coverage of `smd` and `fix_move` on
both `sheetonsheet` and `afm`.

**Where it lives.** Plan Tasks 10 and 12 (`tests/builders/test_sheetonsheet_mlip.py`,
`tests/builders/test_afm_mlip.py`) plus Task 14's `tests/builders/test_family_mlip.py`, which
also carries the related hetero bug fix (then `hetero.py`, now `src/builders/construction/hetero_stack.py`): it registered the Virtual
type *unconditionally*, instead of gating it on
`drive_method == 'virtual_atom'` the way `sheetonsheet/base.py::_register_drive_atom`
does.

**Command.**

```bash
pytest tests/builders/test_sheetonsheet_mlip.py tests/builders/test_afm_mlip.py tests/builders/test_family_mlip.py -v
```

**PASS.** Goldens for both drive methods on both families; no Virtual type
registered when `drive_method='smd'`; a generated `smd` case runs 20 steps under
the appropriate binary.

### M0.6 — (Stretch) MACE through ML-IAP as a hybrid sub-style

**Objective.** `pair_style mace` rejects NULL type maps (ACEsuit #1314), which
is what confines MACE to full-cell runs. The ML-IAP *unified* interface
(`pair_style mliap unified`, Kokkos-GPU) is a different code path whose
hybrid/NULL behaviour is **undocumented**. Cheap to try; if it works, MACE gains
partial-cell coverage and the matrix's row-2/row-3 MACE cells move from **No**
to **Conditional**.

**Command.**

```bash
"$LMP_MACE" -k on g 1 -sf kk -in m0_6_mliap_hybrid.in
```

**PASS.** Runs 20 steps, energies finite, and the mliap-covered layer's energy
matches an equivalent `pair_style mace` monolayer run.

**Any failure mode is a valid FAIL result** — segfault, "unknown keyword",
silently wrong energies. Record it and move on; nothing downstream depends on
this item, and the DeePMD partial-cell path is unaffected.

### M0.7 — (Stretch) `constraint_mode` × MLIP neighbour lists

**Objective.** Spec §3.4: `constraint_mode` values `atom_bonds` and
`com_spring` convert the system to a molecular atom style with explicit bonds.
Whether that interacts badly with MLIP neighbour lists (special-bonds exclusions
silently removing pairs from the MLIP graph) is unvalidated. Until this passes,
Task 12 *rejects* those two constraint modes with an MLIP sheet, in both the
`afm` and `sheetonsheet` builders.

**Command.** Generate a small `sheetonsheet` case with
`constraint_mode = atom_bonds` and an MLIP sheet (temporarily bypassing the
Task-12 validation), then compare step-0 `E_pair` against the same geometry with
`constraint_mode = none`.

**PASS.** `E_pair` identical between the two constraint modes — i.e. the bond
topology does not remove pairs from the MLIP neighbour list. Any difference is a
FAIL and the Task-12 rejection stays permanently.

## Results

Fill in as items are run. `build` is the LAMMPS build used
(`deepmd` = pinned mainline + plugin, `mace` = ACEsuit fork + Kokkos).

Run 2026-08-10 on the local GPU workstation (RTX A4000 16 GB, CUDA 12.5,
driver 580). Builds used: `conda` = conda-forge `lammps 2024.08.29` +
`deepmd-kit 3.1.3` (`cuda129 mpi_mpich`), model `DPA-3.1-3M` frozen at head
`Alex2D`; `src` = locally built LAMMPS `stable_22Jul2025_update1` with
EXTRA-PAIR + PLUGIN (see the "build combination" note below).

`d3bp` rows were run **2026-08-11** on the same workstation with the
**backported** build — LAMMPS `stable_29Aug2024_update3` carrying the
`dispersion/d3` sources from `stable_22Jul2025_update1`
(`scripts/hpc/build_lammps_deepmd_d3.sh`, binary at
`$PREFIX/lammps-deepmd-d3/bin/lmp`), with the `mlip` conda env's deepmd plugin
loaded at runtime. This is the first build on which `pair_style deepmd` and
`pair_style dispersion/d3` coexist, so it is what unblocked M0.2.

| item | build | result | notes |
|---|---|---|---|
| M0.1 deepmd, FULL type map, in hybrid | conda | **PASS** | `pair_style hybrid deepmd <m> lj/cut 11.0` + `pair_coeff * * deepmd Mo S Mo S Mo S`; trilayer 36 atoms, step-0 `E_pair = -256.90309 eV` (-7.14 eV/atom), model on GPU 0 |
| M0.1 deepmd, NULL-partitioned map | conda | **FAIL** | `pair_coeff * * deepmd Mo S NULL NULL NULL NULL` → `ERROR: Pair hybrid sub-style deepmd is not used`. Fails for ONE instance as well as three, so it is NULL that fails, not multi-instance |
| M0.1 deepmd restricted to a type range | conda | **FAIL** | `pair_coeff 1*2 1*2 deepmd Mo S` → `ERROR: deepmd requires that the scale should be set to all atom types, i.e. pair_coeff * *` (`pair_deepmd.cpp:816`) |
| M0.2 deepmd base, no dispersion | d3bp | **PASS** | reference for the D3 contribution: `pair_style deepmd dpa3_alex2d.pth` + `pair_coeff * * Mo S Mo S`; bilayer 24 atoms, step-0 `E_pair = -171.42832 eV` (-7.143 eV/atom) |
| M0.2 deepmd + d3 overlay | d3bp | **PASS** | `pair_style hybrid/overlay deepmd dpa3_alex2d.pth dispersion/d3 bj pbe 16.0 8.0`, both coeff lines `* * <sub> Mo S Mo S`. Step-0 `E_pair = -180.15327 eV`; `compute pair` decomposition `c_edp = -171.42832`, `c_ed3 = -8.7249526`. **D3 contribution = -8.7249526 eV (4.84 % of E_pair).** 20 NVE steps, no segfault, TotEng drift 9e-5 eV. See "RESOLVED" note below |
| M0.2a `dispersion/d3` alone, our emitted string | src | **PASS** | `pair_style dispersion/d3 bj pbe 16.0 8.0` runs; bilayer 24 atoms, `E_pair = -8.7249526 eV` |
| M0.3 mace full-cell base | mace | | step-0 E_pair: |
| M0.3b mace + d3 overlay | mace | | verified d3 argument order: |
| M0.4 virtual atom, mace | mace | | energy offset / max force: |
| M0.4 virtual atom, deepmd | d3bp | **PASS** | type 5 placeholder-mapped to **Mo** (`pair_coeff * * Mo S Mo S Mo`) + `neigh_modify exclude type 5 1..5`. PotEng `-176.07096 eV` vs `-171.42832 eV` for the 24-atom reference → offset **-4.64264 eV, exactly constant**: unchanged to all printed digits after two separate `displace_atoms` of the virtual atom. max&#124;F&#124; on type 5 = 0,0,0 at every step; 20 NVE steps stable |
| M0.4 virtual atom, d3 overlay | src | **PASS** | placeholder-mapped + `neigh_modify exclude type 5 1..5`: PotEng **identical** to the no-virtual run (-8.7249526 eV, not merely a constant offset — a dispersion correction gives an isolated atom exactly zero) and max&#124;F&#124; on the virtual atom = 0,0,0 in x,y,z |
| M0.5 smd / fix_move goldens | n/a | | |
| M0.6 mliap hybrid (stretch) | mace | | |
| M0.7 constraint_mode (stretch) | either | | |

### VERIFIED: the `dispersion/d3` argument order

`PairDispersionD3::settings()` takes exactly 4 arguments in the order
`<damping> <functional> <rthr> <cn_thr>`, requires `units metal`, and squares
both cutoffs internally. Accepted damping tokens are
`original | zero | zerom | bj | bjm` (`original` and `zero` are synonyms).
**This matches `src/core/mlip.py::d3_style_token` exactly** — no change needed.
Our `d3_damping` Literal omits `zero`, which is harmless since `original` is
the same thing.

### RESOLVED: deepmd + dispersion/d3 in one binary (the backport)

**The blocker.** `pair_style dispersion/d3` entered LAMMPS in
`stable_22Jul2025`; the prebuilt deepmd plugin is ABI-locked to LAMMPS
`2024.08.29`, which predates it. Confirmed independently twice:

- pip: loading the wheel's plugin into a 22Jul2025 build fails with
  `undefined symbol: _ZN9LAMMPS_NS5utils11getsyserrorEv`;
- conda: `deepmd-kit 3.1.3` carries the constraint `lammps 2024.08.29`, so the
  solver refuses `lammps 2025.07.22` outright.

**The fix, executed 2026-08-11** (spec v4 §4): rather than move the deepmd side
forward, move the d3 side *back*. `pair_dispersion_d3` is self-contained, so
`scripts/hpc/build_lammps_deepmd_d3.sh` builds the ABI-matching 2024 release
with three files copied in from the 2025 one:

| | |
|---|---|
| host tree | `lammps/lammps @ stable_29Aug2024_update3` (ABI for the plugin) |
| donor tree | `lammps/lammps @ stable_22Jul2025_update1` (d3 sources) |
| files backported | `pair_dispersion_d3.cpp`, `pair_dispersion_d3.h`, `d3_parameters.h` → `src/EXTRA-PAIR/` |
| binary | `$PREFIX/lammps-deepmd-d3/bin/lmp` |

It works, and the numbers are self-validating: under `hybrid/overlay` the
`compute pair` decomposition returns `c_edp = -171.42832` and
`c_ed3 = -8.7249526` — **bit-identical to the two styles run standalone**
(M0.2 base and M0.2a), summing to the reported `E_pair` within 2.6e-6 eV. The
backported d3 is therefore numerically the same code as upstream 2025.

Two caveats the spec did not anticipate, both discovered while executing this:

**1. The copied source does not compile clean against 2024 headers.** It fails
at exactly 11 sites, all of them error-message decoration:
`Error::NOLASTLINE` / `Error::NOPOINTER` (constants added to `error.h` after
2024, telling the newer error printer which argument to blame) and one
`Info::get_pair_coeff_status(lmp)` (a 2025-only diagnostic helper). The build
script applies a mechanical `sed` shim that drops the argument — restoring
stock 2024 behaviour, since `NOPOINTER` is what the 2024 `all(FLERR, fmt, ...)`
overload already implies — and substitutes a literal for the helper. **Every
patched site is inside an `error->all(...)` abort path**, so no executed code
path, energy, or force is affected, and the D3 maths and C6/parameter tables
are byte-for-byte the 2025 originals. Spec v4 §4's "depends only on long-stable
internals" is right about the physics and slightly optimistic about the
diagnostics.

**2. The build MUST be `BUILD_MPI=ON`, even single-GPU.** This is the opposite
of `build_lammps_deepmd.sh`, and it is not a preference. The deepmd plugin
leaves 20 `MPI_*` symbols undefined and expects the host executable to supply
them. A no-MPI LAMMPS supplies MPI from `src/STUBS/mpi.cpp`, which is **C++** —
its symbols are mangled (`_Z13MPI_AllgatherPviiS_iii`) and do **not** satisfy
the plugin's C-ABI `MPI_Allgather`. Build against the same MPI the plugin was
built against (conda-forge deepmd-kit ⇒ `mpi_mpich`; on this host, the `mlip`
env's own MPICH — which also sidesteps the broken system MPICH).

**Both failure modes above look identical at runtime**, and both look like a
misconfigured `LAMMPS_PLUGIN_PATH` rather than a build problem:

```
Open of .../dpplugin.so failed: undefined symbol: MPI_Allgather
Loaded 0 plugins from .../deepmd_lmp
ERROR: Unrecognized pair style 'deepmd'
```

The third variant of the same symptom is a missing `LD_LIBRARY_PATH`
(`libabsl_log_flags.so.2601.0.0: cannot open shared object file`) — the plugin
pulls in the conda env's own libabsl/TensorFlow runtime. **`Loaded 1 plugins`
in the log is the only proof the plugin is actually live**; the build script
asserts on it with a `plugin list` probe deck rather than trusting `lmp -h`
(which cannot show a runtime plugin at all).

### Running the backported binary

```bash
export LMP=$HOME/opt/lammps-deepmd-d3/bin/lmp
export LAMMPS_PLUGIN_PATH=/home/mv923/miniconda3/envs/mlip/lib/deepmd_lmp
export LD_LIBRARY_PATH=/home/mv923/miniconda3/envs/mlip/lib
env -u PYTHONPATH "$LMP" -in deck.in       # env -u: see the PYTHONPATH note below
```

`env -u PYTHONPATH` is needed for anything launched from the repo: the user's
profile points `PYTHONPATH` at a *different* FrictionSim2D checkout, which
shadows the editable install.

The four decks behind the `d3bp` rows are kept at `/home/mv923/opt/m0/phaseD/`
(`m0_2_base.in`, `m0_2_deepmd_overlay_d3.in`, `m0_4_dp_ref.in`, `m0_4_dp_virt.in`)
with their logs, driven by `run_all.sh`. The M0.2 overlay deck is:

```
units metal
atom_style atomic
read_data bilayer_mos2.data
pair_style hybrid/overlay deepmd dpa3_alex2d.pth dispersion/d3 bj pbe 16.0 8.0
pair_coeff * * deepmd Mo S Mo S
pair_coeff * * dispersion/d3 Mo S Mo S
compute edp all pair deepmd
compute ed3 all pair dispersion/d3
thermo_style custom step pe c_edp c_ed3 etotal
```

`compute pair <sub-style>` is what makes the D3 contribution separable — do not
infer it by differencing two runs when one deck can report both terms.

### VERIFIED: the virtual atom survives an MLIP (M0.4, DeePMD half)

The placeholder+exclusion mechanism holds for DeePMD, not just for D3, so
`drive_method = virtual_atom` stays available for full-cell MLIP runs. The
distinction that matters:

- under **D3**, the offset is *zero* — a dispersion correction gives an
  isolated atom exactly nothing (`-8.7249526 eV` either way);
- under **DeePMD**, the offset is **-4.64264 eV**, the isolated-atom energy of
  the Mo placeholder in the `Alex2D` head. Non-zero is expected and fine; what
  matters is that it is *constant*, which was tested by displacing the virtual
  atom twice and confirming PotEng did not move at all.

So an MLIP virtual atom shifts the absolute energy by a constant. Forces,
force differences, and every friction observable are unaffected — but absolute
`E_pair` from a `virtual_atom` run is **not** comparable to a `smd`/`fix_move`
run of the same system without subtracting that constant.

The placeholder element is a free choice among elements the model knows; the
type-5 mass in `bilayer_plus_virtual.data` stays 1.0 (H) and is irrelevant,
since an excluded atom feels exactly zero force regardless of mass.

### TRAP: `pair_coeff * * deepmd` with no element list is silently wrong

Omitting the type map does not error. LAMMPS types are then mapped to the
model's first N types — in a 118-element foundation model that is H, He, Li,
Be, B, C. The same trilayer gives `E_pair = -61.89 eV` instead of `-256.90 eV`.
Always pass the explicit element list.

## On completion

Milestone 0 results are not just notes — they are inputs to the code. When the
table above is filled in:

1. **Update the capability matrix** in
   [`docs/superpowers/specs/2026-08-06-mlip-support-design.md`](../docs/superpowers/specs/2026-08-06-mlip-support-design.md):
   every *Conditional* cell becomes **Yes** or **No**, and the footnotes that
   justified "Conditional" are rewritten to cite this runbook's result instead.
2. **Update the `supports_*` flags** in `src/core/mlip.py` — `supports_null`,
   `supports_overlay`, and the virtual-atom support implied by M0.4 — so
   validation rejects at config time whatever Milestone 0 proved impossible.
   The flags are the single source of truth the builders and validators read;
   do not encode a Milestone-0 outcome anywhere else.
3. **Record the verified `dispersion/d3` argument order** in the registry's
   `d3_style_token` / `d3_tokens` translation, **per build** — the two builds
   are allowed to need different tokens, which is the reason `d3_damping` is a
   canonical name rather than a literal. Re-run the per-build D3 golden tests
   afterwards (see `scripts/hpc/README.md`).
4. If M0.4 failed, add the `smd`/`fix_move` fallback requirement to the config
   validation and drop the §1.4 placeholder/exclusion emission.
5. Commit as `Record Milestone 0 results`.

## Model acquisition

Foundation models are downloaded and converted once; the converted artefacts are
what LAMMPS reads. None of this needs a GPU.

### MACE — `mace-mp-0b3-medium`

```bash
source "$PREFIX/mace-venv/bin/activate"   # has mace-torch

# 1. Download the pinned MACE-MP release checkpoint.
python - <<'PY'
from mace.calculators.foundations_models import mace_mp
# Downloads into ~/.cache/mace and prints the resolved checkpoint path.
calc = mace_mp(model="medium", default_dtype="float64", device="cpu")
print(calc.model_path if hasattr(calc, "model_path") else calc)
PY

# 2. Convert the checkpoint into the LAMMPS-consumable form.
mace_create_lammps_model <path-to>/mace-mp-0b3-medium.model
```

`mace_create_lammps_model` writes `<name>.model-lammps.pt` next to its input —
that is the file `pair_coeff * * mace <file> ...` names. It defaults to
**float64**; keep it, and note the precision in the results table, because
float32 conversions change energies at the meV level and would break M0.4's
"constant offset" comparison if mixed between runs.

Copy or symlink the result into `scripts/milestone0/` as `mace.model-lammps.pt`.

### DeePMD — a DPA model

DeePMD needs a **frozen** model; a training checkpoint will not load. DPA
models are multi-head, so the freeze command must name the head to extract.

```bash
source "$PREFIX/deepmd-venv/bin/activate"   # has deepmd-kit[torch]==3.0.2

# 1. Fetch the DPA checkpoint (see documentation/mlip.md for the pinned model
#    and its URL; keep the exact version pinned there, not here).

# 2. Freeze the chosen head into a LAMMPS-loadable .pth.
dp --pt freeze -c <checkpoint-dir> -o frozen_model.pth --head <head-name>
```

`DP_ENABLE_PYTORCH=1` must have been set when deepmd-kit was installed (the
build script does this) or the PyTorch custom OPs are missing and the freeze
silently produces a model LAMMPS cannot run.

Copy the result into `scripts/milestone0/` as `frozen_model.pth`.

> A published MoS2 DeePMD model is acceptable **only** as a code-path test
> fallback for this runbook. It is not a scientific substitute for a DPA
> foundation model, and results obtained with it must not be quoted as physics.

Never write `pair_style deepmd model1.pth model2.pth` — two model paths switch
DeePMD into model-deviation mode rather than giving you two instances. Multiple
instances are expressed through `pair_style hybrid` with indexed sub-styles, as
in M0.1.

### MACE for ML-IAP (M0.6 only)

```bash
mace_create_lammps_model --format=mliap <path-to>/mace-mp-0b3-medium.model
```

writes the `mace-mliap_lammps.pt` that `pair_style mliap unified` loads. This
requires the MACE build's `ML-IAP` + `ML-SNAP` + `PYTHON` packages with
`MLIAP_ENABLE_PYTHON=ON`, which `build_lammps_mace.sh` already enables — no
rebuild needed.

---

## Post-M0: production-scale findings (generate → run, 2026-08-11)

The Milestone-0 decks above are hand-written. Running **FrictionSim2D's own
generated output** through the same binary exposed three further problems that
no component-level test could catch, because each artifact was individually
correct and only their composition was broken.

### F1 — `pair_style zero` surrogate leaked `Pair Coeffs` into the structure (FIXED)

`pair_style zero` has per-type coefficients (the classical many-body styles do
not), so LAMMPS `write_data` serialised a `Pair Coeffs # zero` block into the
stacked data file. Production scripts `read_data` **before** they `include` the
settings file that defines `pair_style`, so LAMMPS aborted:

```
ERROR: Must define pair_style before Pair Coeffs (read_data.cpp:782)
```

Fixed by `write_data ... nocoeff` in `src/templates/common/stack_layers.lmp`,
pinned by `tests/builders/test_components_mlip.py::test_mlip_stacking_data_file_carries_no_pair_coeffs`.

### F2 — `comm_style tiled` is incompatible with `pair_deepmd` (OPEN)

A generated run dies in `pair_deepmd.cpp:253` trying to allocate ~780 GB:

```
DeePMD-kit PyTorch backend error: ... DefaultCPUAllocator: can't allocate
memory: you tried to allocate 778824196096 bytes
```

Bisected properly, and it is **none of the obvious suspects**:

| Hypothesis | Verdict |
|---|---|
| System size | **No** — fails at 270, 432 and 810 atoms alike |
| D3 overlay | **No** — fails identically with the overlay removed |
| Model on CPU not GPU | **No** — `load model from: ... to gpu 0` in every run |
| `balance rcb` | **No** — removing `balance` alone, keeping tiled, still fails (558731354112 bytes) |
| `comm_style tiled` | **YES** — removing tiled alone, the identical 270-atom deck runs: `Loop time of 13.97` |

It is **not** a GPU problem: the model loads to the device in every run
(`load model from: ... to gpu 0`), the failing allocation is
`DefaultCPUAllocator` (host side), and the size varies run to run
(558/778/831 GB) — the signature of a garbage length, not real demand.

Mechanism, as far as it can be established from outside deepmd-kit:
`comm_style tiled` is a different communication path — irregular RCB tiles
rather than a regular 3D grid, variable neighbour counts, and a documented
restriction that the comm cutoff **cannot exceed the box length** (which is
exactly F3's error; `brick` instead does multiple ghost shells).
`pair_deepmd` builds its own neighbour structures from LAMMPS's ghost region
and appears only to have been exercised against the brick layout.

This is a `pair_deepmd` limitation, NOT a property of MLIPs: nothing about a
machine-learned potential requires brick decomposition. Worth an upstream
report.

**Workaround: do not emit `comm_style tiled` for an MLIP sheet.** The cost is
nil at our design point -- tiled decomposition and RCB balancing exist to
spread irregular work over many MPI ranks, and both frameworks run
single-rank-per-GPU (`LauncherSpec.mpi_ranks == 1`). Proper fix is a builder
change suppressing it for MLIP runs.

### F3 — D3's 16 A cutoff demands a box larger than ~16.3 A

With `d3_cutoff: 16.0` (+0.3 skin) the communication cutoff is 16.3 A, so any
sheet whose periodic box is smaller fails hard:

```
ERROR: Communication cutoff for comm_style tiled cannot exceed periodic box length
```

Measured: a `x = y = 15` sheet builds a 15.9 x 16.5 A box and fails; `x = y = 20`
(19.1 x 22.1 A) clears it. Small screening cells therefore need a reduced
`d3_cutoff`, or a larger footprint. This deserves config-time validation.

### F4 — the backported build needs more packages

`fix rigid/nve` (used by the sheet-on-sheet template) requires RIGID, which the
first backported build omitted. `scripts/hpc/build_lammps_deepmd_d3.sh` now also
enables `PKG_RIGID`, `PKG_KSPACE` and `PKG_EXTRA-FIX`.
