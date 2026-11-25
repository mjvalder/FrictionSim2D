# Installation

Environment setup, optional extras, verification, and troubleshooting.

## Requirements

- Python 3.9+
- LAMMPS executable available in your environment
- Atomsk executable available in your environment (optional)

Optional features require extra dependencies:

| Extra | Adds | Needs |
|---|---|---|
| `aiida` | the local provenance store (`aiida-core`) | Python 3.9+ |
| `plotting` | `matplotlib`, `seaborn`, `scipy` | Python 3.9+ |
| `gui` | the desktop builder (`ovito`, `PySide6`) | see [gui.md](gui.md) |
| `agent` | the GUI [Assistant](assistant.md) (`pydantic-ai`, `huggingface_hub`) | Python 3.11+ |
| `dev` | `pytest`, `pytest-cov`, `pylint`, `pytest-qt` | |

## Install from source

From the project root:

```bash
pip install -e .
```

Install optional extras as needed. Quote the extras so the shell does not glob the
brackets:

```bash
pip install -e ".[aiida]"
pip install -e ".[plotting]"
pip install -e ".[gui]"
pip install -e ".[agent]"
pip install -e ".[all]"
```

## Verify the installation

```bash
FrictionSim2D --help
FrictionSim2D --version
```

`FrictionSim2D gui --check` verifies the GUI environment separately (see
[gui.md](gui.md)). Then verify external binaries used by workflows:

```bash
lmp -h
```

If you use Atomsk-based geometry operations, also verify:

```bash
atomsk --version
```

## Materials and potentials

The material (CIF) and potential library ships inside the package, in
`frictionsim2d/data/`, so an installed copy finds it without the repository.
The package carries one amorphous Si block; the verified pack of further
pre-quenched amorphous blocks is optional:

```bash
FrictionSim2D fetch materials            # --remove deletes it again
```

It goes to `$XDG_DATA_HOME/FrictionSim2D/materials-pack` and is searched after
the packaged files. The assistant's fine-tuned models are installed the same way
(`FrictionSim2D fetch model`, see [assistant.md](assistant.md)).

## AiiDA setup (optional)

After installing AiiDA dependencies:

```bash
FrictionSim2D aiida status
FrictionSim2D aiida setup
```

There is no SSH-connected remote-submission mode — the HPC blocks direct
inbound AiiDA connections. Two workflows cover a remote cluster instead:

- **remote-manual**: generate locally, `rsync` the tree out, submit the
  generated PBS/Slurm scripts on the cluster by hand, `rsync` the results
  back, then ingest with a plain `FrictionSim2D aiida import` — nothing
  AiiDA-specific ever runs on the cluster. See
  [hpc_jobs.md](hpc_jobs.md#getting-results-back-into-aiida-remote-manual).
- **hpc-direct**: install and run `FrictionSim2D aiida setup` *inside the
  HPC's own conda environment*, ingest there, then `FrictionSim2D aiida
  export <label>` to build a portable archive and bring it home with
  `FrictionSim2D aiida import-archive` on your workstation.

See [aiida.md](aiida.md) for both in full.

## Troubleshooting

`FrictionSim2D: command not found`

- Check that your environment is active.
- Re-run `pip install -e .` from repository root.

`lmp` missing

- Install LAMMPS in the same environment used to run FrictionSim2D.

`atomsk` missing

- Atomsk is optional. Install it only if you plan to use Atomsk-based operations.

AiiDA commands abort immediately

- Install AiiDA dependencies (`pip install -e ".[aiida]"`), then rerun `FrictionSim2D aiida setup`.

## Related docs

- [overview.md](overview.md)
- [commands.md](commands.md)
- [aiida.md](aiida.md)
