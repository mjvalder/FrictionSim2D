# FrictionSim2D materials pack (upload staging)

This folder stages the materials pack that `FrictionSim2D fetch materials`
installs. It is published as the Hugging Face dataset repo
[`mjvalder/fs2d-materials`](https://huggingface.co/datasets/mjvalder/fs2d-materials).

## Contents

Pre-quenched amorphous blocks, in the same flat format as the user library
(`$XDG_DATA_HOME/FrictionSim2D/amorphous`):

- `amor_<mat>__<key8>.lmp`: the LAMMPS data file;
- `amor_<mat>__<key8>.lmp.json`: its sidecar, which holds the quench conditions,
  the potential sha256 and `block_sha256`;
- `manifest.json`: every file with its sha256 and size.

The amorphous Si block (`amor_Si__021d81ab`) ships inside the package
(`frictionsim2d/data/materials`) and is not part of the pack.

Material lookups try roots in this order: packaged, pack, user library.

## Publishing a block

1. Make the block and import it: `FrictionSim2D run quench ...`, run it, then
   `FrictionSim2D quench import ROOT`. It lands in the user library.
2. Copy `amor_<mat>__<key8>.lmp` and its `.lmp.json` from
   `$XDG_DATA_HOME/FrictionSim2D/amorphous/` into this folder.
3. `python tools/materials_pack/make_manifest.py`. It refuses a block without
   its sidecar, or one whose `block_sha256` does not match.
4. Upload every file in this folder to the dataset repo, for example
   `huggingface-cli upload mjvalder/fs2d-materials tools/materials_pack . --repo-type dataset`.

`fetch materials` downloads into a staging folder and checks every size and
sha256 against the manifest. It replaces the installed pack only when all of
them match.

Staged blocks and the manifest are git-ignored (see `.gitignore`), so they are
never committed.
