# MLIP models — how to get them

Unlike every other directory under `frictionsim2d/data/potentials/`, **this one ships no
model files.** Classical potentials here are KB-scale text you can read and
diff; MLIP foundation models are 18 MB–120 MB of third-party binary weights with
their own licences and release channels. Putting them in the repository would
add hundreds of megabytes to every clone and pin you to whatever version was
current the day it was committed.

So this directory ships the **recipe** instead. One command per framework:

```bash
# from the repository root
./scripts/fetch_mlip_models.sh deepmd                 # -> ~/opt/models/dpa3_alex2d.pth
./scripts/fetch_mlip_models.sh mace --elements Mo S   # -> ~/opt/models/mace-mpa-0-medium-Mo-S.json
```

Then point a config at the result:

```ini
[2D]
pot_type = deepmd
pot_path = /home/you/opt/models/dpa3_alex2d.pth
dispersion = d3
lat_c = 6.112
```

`examples/mlip/*.ini` are working configs that need only the `pot_path` line
changed. Full background, traps and the capability matrix:
[`documentation/mlip.md`](../../../documentation/mlip.md).

---

## The one thing that differs between the frameworks

**DeePMD: one file covers everything. MACE: one download, one extraction per
element set.**

| | DeePMD | MACE |
|---|---|---|
| What you download | `DPA-3.1-3M.pt` checkpoint, ~47 MB | `mace-mpa-0-medium.model`, ~79 MB |
| Elements in the download | 118 (all of them) | 89 (H–Bi, Ac–Pu) |
| What LAMMPS reads | `dpa3_alex2d.pth`, ~18 MB | `mace-mpa-0-medium-<els>.json`, ~120 MB |
| Elements in *that* file | **118 — reusable for every material** | **only the ones you asked for** |
| New material ⇒ ? | nothing to do | **re-extract** (no new download) |

Two prerequisites for the MACE side, both one-time:

- `symmetrix`'s `pyproject.toml` lives in the repo's `symmetrix/` **subdirectory**,
  so it installs as
  `pip install "git+https://github.com/wcwitt/symmetrix.git#subdirectory=symmetrix"`.
  A plain `git+…` URL fails.
- It is a scikit-build-core package with a pybind11 extension, so it **compiles
  on install**: you need `cmake >= 3.27` and a C++ compiler. The machine that
  *runs* LAMMPS needs none of this — only the machine that extracts the `.json`.
  If you already ran `scripts/hpc/build_lammps_mace.sh` there is a clone on disk;
  point `SYMMETRIX_SRC` at it to install from there instead of re-cloning.

The MACE `.json` is *derived locally* from the checkpoint you already have —
`symmetrix_extract_mace --chemical-symbols Mo S` writes a model whose
`atomic_numbers` field is literally `[16, 42]`. Adding WS₂ means running the
extractor again with `--elements W S`; it does **not** mean downloading another
foundation model. The element suffix is applied for you (sorted, so argument
order does not matter: `--elements Mo S` and `--elements S Mo` both give
`mace-mpa-0-medium-Mo-S.json`), which keeps element sets from colliding in the
campaign's shared `potentials/` store — that store is keyed by filename.

If a MACE checkpoint is multihead, `MACE_HEAD=<name>` selects which head to keep;
the default is the extractor's own (the first non-PT head).

## Which DeePMD model, and why not the newest

`fetch_mlip_models.sh` defaults to **DPA-3.1-3M** frozen at the **`Alex2D`**
head. Both halves of that are deliberate.

`dp pretrained download` currently also offers `DPA-3.2-5M`, `DPA-3.3-1M`,
`DPA3-Omol-Large` and a whole `DPA4-*` generation, and you can pass any of them
with `--model`. Before you do, know what you are re-opening:

1. **The head must exist.** DPA-3.1-3M is multitask with 31 heads; `Alex2D` is
   the Alexandria 2D-materials head, which is the right training domain for
   layered systems. A newer checkpoint is not guaranteed to carry a head by that
   name — and silently picking a different one changes the physics, not just the
   accuracy. Check with `--list-heads` before committing to a model.
2. **`lat_c` is model-specific.** It is a *mandatory* config value and the
   examples ship `lat_c = 6.112`, which is the measured DPA-3.1/`Alex2D` + D3
   equilibrium. A different model has a different equilibrium separation, so
   switching models means re-running
   `scripts/mlip_bilayer_benchmark.py` and updating every config. Using the old
   number with a new model is the "dome plow" failure mode.
3. **deepmd-kit and LAMMPS are ABI-coupled.** `scripts/hpc/build_lammps_deepmd.sh`
   pins deepmd-kit 3.0.2, and deepmd-kit pins a LAMMPS release — which is the
   entire reason `build_lammps_deepmd_d3.sh` exists as a backport. A model that
   needs a newer deepmd-kit can cascade into rebuilding that backport.

None of this makes newer models wrong — it makes them a *measurement exercise*,
not a config edit. The published numbers in `documentation/mlip.md` §5 all rest
on DPA-3.1-3M/`Alex2D`, so that is what the defaults reproduce.
