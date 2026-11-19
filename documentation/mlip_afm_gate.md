# The AFM MLIP physics gate (v6 §6)

**Status: Gates 1–3 PASS. Gate 4 is recorded as a finding, not a pass.**
Run 2026-09-09 on the local GPU workstation (RTX A4000, 16 GB).

v6 made a full-cell MLIP legal in a *tip-bearing* cell (AFM, `pes-tip`,
`indent`). This file is the evidence that the capability is real, because the
failure mode of this stack is **silent**: LAMMPS maps unknown types onto the
model's first N elements and returns a plausible, wrong energy (v4 evidence E7).
An exit code proves nothing here, so nothing is recorded as an AFM MLIP result
until the gates below have been run against the actual generated tree.

The reference cell throughout is `examples/mlip/afm_mlip_full.ini` with absolute
model paths: 3-layer h-MoS2 (30×30 Å), crystalline lattice-matched Si substrate,
r = 15 Å crystalline Si tip. **1512 atoms** — 702 Si (576 substrate + 126 tip),
270 Mo, 540 S — in 13 atom types, the 13th being the virtual drive atom, which
carries zero atoms at this stage and is excluded from the neighbour list.

## Gate 1 — deck shape: PASS (both frameworks)

Generated with `run afm` on a real tree, not a unit-test fixture. Both
frameworks emit exactly one `pair_coeff * *` for the model, spanning all 13
types, with **no `NULL` and no inter-component `lj/cut`**:

```
pair_style hybrid/overlay symmetrix/mace dispersion/d3 bj pbe 16.0 8.0
pair_coeff * * symmetrix/mace <model>.json Si Si Si Si Si Si Mo Mo Mo S S S Mo
pair_coeff * * dispersion/d3                Si Si Si Si Si Si Mo Mo Mo S S S Mo
```

(DeePMD is identical with `deepmd <model>.pth` and no path on the coeff line.)
Six Si types rather than two because Langevin splits each of the substrate and
tip into mobile/fix/thermo regions; three Mo and three S because the sheet is
three layers.

**This gate did not pass when it was first claimed.** The `system.in.settings`
half was verified on 2026-09-04, but the *decks* were unrunnable under MACE:
`mlip_deck_directives` (`atom_modify map yes`, `newton on`) and `mlip_sheet`
(which suppresses `comm_style tiled`) were never wired into the AFM, hetero or
flake builders. Both keys are rendered behind Jinja's `| default(...)`, so a
missing key renders *nothing* and the build still reports success. Fixed
2026-09-09; see "What this gate caught" below.

## Gate 2 — Si in isolation: PASS (both models)

The out-of-domain risk this catches is specific: an AFM asperity is bulk Si,
and a 2D-materials head has never seen that phase. Lattice constant is the
primary metric because, unlike a cohesive energy, it does not depend on the
model's atomic energy zero. 64-atom diamond cell, `run 0`, parabolic fit of the
three points bracketing the minimum.

| Model | a₀ (Å) | vs PBE 5.468 | vs expt 5.431 | E_coh (eV/atom) |
|---|---|---|---|---|
| MACE-MPA-0 medium (Mo,S,Si extraction) | **5.4675** | −0.01 % | +0.67 % | 4.588 |
| DPA-3.2, OMat24 head | **5.4760** | +0.15 % | +0.82 % | 4.370 |

Both reproduce the PBE Si lattice constant to well under 0.2 %. Reference
cohesive energies: PBE ≈ 4.5–4.6 eV/atom, experiment 4.63 eV/atom — MACE lands
inside that range; DPA-3.2/OMat24 is ~4 % low.

**Treat the E_coh column as the weaker of the two numbers.** It is
`E_isolated_atom − E_bulk_per_atom`, and an isolated atom in vacuum is itself
out of domain for a model trained on bulk solids, so part of that 4 % is the
reference, not the bulk description. The lattice constant carries this gate.

## Gate 3 — energy agreement: PASS

The strongest of the four, and the one that directly targets v4 E7. Step-0
energy of the generated cell computed two ways that share nothing but the
geometry and the underlying model:

| Path | Code | Model file | E_pair (eV) |
|---|---|---|---|
| LAMMPS | `pair_symmetrix/mace` (C++) | `mace-mpa-0-medium-Mo-S-Si.json` | −8843.15649620 |
| ASE | `mace-torch` 0.3.16 (PyTorch) | `mace-mpa-0-medium.model` | −8843.15675071 |

**Δ = 2.55 × 10⁻⁴ eV over 1512 atoms = 1.7 × 10⁻⁷ eV/atom.** Different
implementation, different serialization, same answer. This is what makes the
emitted type map trustworthy: ASE was handed the map independently
(types 1–6 → Si, 7–9 → Mo, 10–12 → S) and recovered the composition the builder
intended — Si 702, Mo 270, S 540.

Run MLIP-only (no D3 overlay) on purpose: the question is whether *our type map*
reproduces the model, and adding a second pair style would only blur it.

## Gate 4 — cross-framework: recorded, NOT a pass

MACE-MPA-0 vs the DPA-3.2 **MPtrj** head — the pairing chosen because both are
trained on MPtrj, so a disagreement is architecture, not training data.
Compared on **forces**, not energies: each model has its own atomic energy zero,
so raw `E_pair` is not comparable across frameworks (a −8843.16 vs −8830.12 eV
gap is mostly reference shift and means nothing on its own).

Same 1512-atom cell, as generated (unrelaxed):

| Quantity | Value |
|---|---|
| RMS \|F\|, MACE-MPA-0 | 0.6707 eV/Å |
| RMS \|F\|, DPA-3.2 MPtrj | 0.5648 eV/Å |
| RMS componentwise difference | **0.1720 eV/Å** |
| Relative RMS difference | **44.4 %** |
| Pearson *r* (all components) | 0.8977 |
| mean cos∠(F_MACE, F_DPA) | 0.5785 (median 0.8978) |

Per element: Si 52.2 %, Mo 41.9 %, S 41.7 % relative RMS.

### Checked again on a relaxed geometry — the disagreement does not soften

The table above is an as-built cell, where forces are large and cross-model
disagreement is easiest. To test whether 44 % was an artifact of that strain,
the cell was relaxed under MACE (CG, 400 steps, E: −8843.16 → **−9134.63 eV**,
so the as-built geometry was indeed heavily strained) and both models
re-evaluated on the **same relaxed coordinates**:

| Quantity | as-built | MACE-relaxed |
|---|---|---|
| RMS \|F\|, MACE-MPA-0 | 0.6707 | **0.0216** eV/Å |
| RMS \|F\|, DPA-3.2 MPtrj | 0.5648 | **0.2708** eV/Å |
| **RMS componentwise difference** | **0.1720** | **0.1564** eV/Å |
| Pearson *r* | 0.8977 | **0.0309** |
| mean cos∠ | 0.5785 | 0.0922 |

**The absolute disagreement is essentially unchanged — 0.172 → 0.156 eV/Å.** It
was never a strain artifact. What changes is only the denominator: MACE's own
residual force falls to 0.022 eV/Å at its minimum while DPA-3.2 still reports
0.271 eV/Å on that same structure, twelve times larger. The two models **do not
share a minimum for this cell**, and near MACE's minimum their forces are
uncorrelated (*r* = 0.03).

Ignore the 1253 % "relative" figure this produces — it divides by a base that
is near zero by construction. The number to carry is the absolute 0.156 eV/Å,
and the fact that it is geometry-independent.

For friction work this is the material point: the quantities of interest are
interfacial forces and corrugation barriers, and a 0.16 eV/Å model-dependent
force difference is not a small perturbation on those.

**This is a finding, and it is not small.** Two architectures trained on the
same data disagree on ~44 % of the force magnitude on this geometry. The mean
cosine (0.58) sitting far below the median (0.90) says the disagreement is
concentrated in a minority of atoms whose force directions differ sharply, not
spread evenly — consistent with small-magnitude forces whose direction is
poorly determined, but not proven to be only that.

No tolerance is declared passed. v6 §6 asks for agreement "to within a stated
tolerance"; 44 % is not an agreement, so **Gate 4 is recorded as open**. Do not
report an AFM MLIP friction number as framework-independent on this evidence.

## Bridging measurement (v6 §8 cross-level risk)

v6 §8 asks for one cell run under both the sheet campaign's head and the
MPtrj head **before any sheet-vs-AFM comparison**, so that a later difference
can be attributed to physics rather than to the model swap. Run on the M0
trilayer MoS2 reference cell (36 atoms, sheet-only — Alex2D is a 2D-materials
head and is not valid for a tip-bearing cell, v6 §2).

Energies reproduce `/home/mv923/opt/models/README.md` exactly
(`dpa3_alex2d` = −256.89789 eV), which confirms the invocation matches the
established one. Compared on forces:

| Pair | RMS diff (eV/Å) | relative | Pearson *r* |
|---|---|---|---|
| `dpa3_alex2d` → `dpa32_alex2d` (checkpoint version, C1) | 0.0437 | 24.8 % | 0.978 |
| `dpa32_alex2d` → `dpa32_omat24` (head, C2) | 0.1306 | 66.7 % | 0.993 |
| **`dpa32_alex2d` → `dpa32_mptrj` (the bridge)** | **0.0840** | **42.8 %** | 0.983 |
| `dpa3_alex2d` → `dpa32_mptrj` (v5 campaign model → bridge) | 0.1220 | 69.2 % | 0.933 |

**The conclusion is a caution, not a green light.** Swapping the head alone
moves the forces on an identical geometry by 43 % (and 69 % from the model the
v5 sheet campaign actually used). A sheet-vs-AFM difference measured across
these heads is therefore **not** attributable to physics — the model swap is
the same size as, or larger than, any effect being looked for. Either run both
levels on the same head, or carry this offset explicitly.

Two caveats on the numbers. This cell is small (36 atoms), symmetric, and near
equilibrium, so forces are small (RMS 0.31–0.56 eV/Å) and a relative percentage
divides by a small base. The `mean cos` values land on exact ninths
(1.0000, 0.7778, 0.5556, 0.3333), which says the per-atom force directions are
symmetry-locked to ±1 and the mean is just counting how many flipped — read it
as "a third of atoms reverse direction", not as an angular average.

## What this gate caught

Running it end to end found three defects that the unit suite did not, all the
same shape — a Jinja context key that no builder supplied, rendered behind
`| default(...)` so its absence was silent:

1. **`mlip_deck_directives` missing from the AFM family.** Every generated
   AFM / `pes-tip` / `indent` MACE deck lacked `atom_modify map yes`, and
   `pair_symmetrix_mace.cpp:172` aborts without it.
2. **`mlip_sheet` missing everywhere except sheet-on-sheet.** Four templates
   (`afm/system_init.lmp`, `afm/slide.lmp`, `afm/indent.lmp`,
   `afm_freestanding/pes_scan.lmp`) had no guard at all, and three more had the
   guard but no builder supplying the key. Under `pair_deepmd`, the
   `comm_style tiled` this suppresses causes a bogus ghost count and a
   hundreds-of-GB allocation attempt.
3. **`afm/slide.lmp` — the AFM *sliding* deck — had neither.** It was missing
   from the hand-maintained template list that the first two fixes worked from.
   `tests/builders/test_mlip_deck_directives.py` now **derives** that list from the
   templates themselves, with an explicit, justified exemption set for the
   generation-time surrogate templates, so a new template joins the tests by
   existing rather than by being remembered.

## Operational notes

**DPA-3.2 does not fit this cell on a 16 GB GPU.** 1512 atoms OOMed at ~14.3 GB
(`pair_deepmd.cpp:373`); the Gate 4 DeePMD numbers were produced on CPU
(`CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=16`). Size GPU jobs accordingly — this
is a reference cell, and production AFM cells are larger.

**The MACE-capable LAMMPS is not the conda one.** `symmetrix/mace` and
`dispersion/d3` live in `/home/mv923/opt/lammps-mace/build-gpu/lmp`, and it must
find **its own** `liblammps.so.0` first or it dies with
`undefined symbol: lammps_plugin_finalize`:

```bash
B=/home/mv923/opt/lammps-mace/build-gpu
CE=/home/mv923/miniconda3/envs/mlip
env -u PYTHONPATH LD_LIBRARY_PATH=$B:$CE/lib $B/lmp -in deck.in
```

`env -u PYTHONPATH` is required — the login profile points `PYTHONPATH` at a
different FrictionSim2D checkout. DeePMD instead uses the conda env's own `lmp`
plus its matching plugin dir, exactly as `/home/mv923/opt/models/README.md`
describes.

## Reproducing

Decks, dumps and the comparison scripts are under the session scratchpad; the
two analysis scripts are small enough to re-derive from this file. The generated
trees come from `examples/mlip/afm_mlip_full.ini` with `pot_path` made absolute
(and `pot_type` switched to `mace` with the Mo-S-Si extraction for the MACE
half).
