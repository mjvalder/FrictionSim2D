# Provenance

How FrictionSim2D records input provenance for reproducibility.

## Design goals

- Keep enough metadata to reconstruct how a simulation was built.
- Track which source files were used by each component.
- Keep manifest data easy to consume from downstream tools (including AiiDA flows).

## Current behaviour

Provenance is handled by `SimulationBase` methods in `src/core/simulation_base.py`.

When source files are added:

- CIF files are copied into `provenance/cif/`
- Potential files are copied into `provenance/potentials/`
- Metadata entries are written into `provenance/manifest.json`

Each manifest record stores fields like:

- `filename`
- `original_path`
- `stored_path`
- `category`
- `components`
- `checksum`
- `added_at`

## Typical provenance tree

```text
<simulation>/
  provenance/
    manifest.json
    config.json
    cif/
      <files>.cif
    potentials/
      <potential files>
```

AFM simulations may also include layer-resolved directories (`L1`, `L2`, ...), but provenance tracking remains centered around each simulation path.

## Why checksums matter

Checksums let you:

- confirm file integrity after transfer
- detect accidental replacement of source files
- compare whether two runs used identical assets even if path names changed

## Consumption by workflows

AiiDA integration and postprocessing tools can read provenance contents to map simulation outputs back to source inputs.

## Limitations and trade-offs

- Current implementation copies files rather than creating symlinks, prioritizing portability.
- If very large potential files are used repeatedly, storage usage can increase.
- Manifest schema is simple JSON; there is no strict external schema file yet.

## Related docs

- [overview.md](overview.md)
- [aiida.md](aiida.md)
- [python_api.md](python_api.md)
