# Installation

Environment setup, optional extras, verification, and troubleshooting.

## Requirements

- Python 3.9+
- LAMMPS executable available in your environment
- Atomsk executable available in your environment (optional)

Optional features require extra dependencies:

- AiiDA workflows: `aiida-core`
- Database commands: `psycopg2-binary`
- Plotting: `matplotlib`, `seaborn`, `scipy`
- API server: `fastapi`, `uvicorn`, `httpx`

## Install from source

From the project root:

```bash
pip install -e .
```

Install optional extras as needed. Quote the extras so the shell does not glob the
brackets:

```bash
pip install -e ".[aiida]"
pip install -e ".[db]"
pip install -e ".[plotting]"
pip install -e ".[api]"
pip install -e ".[all]"
```

## Verify the installation

```bash
FrictionSim2D --help
FrictionSim2D --version
```

Then verify external binaries used by workflows:

```bash
lmp -h
```

If you use Atomsk-based geometry operations, also verify:

```bash
atomsk --version
```

## AiiDA setup (optional)

After installing AiiDA dependencies:

```bash
FrictionSim2D aiida status
FrictionSim2D aiida setup
```

If your workflows target a remote machine, configure the `aiida` section in settings and use:

```bash
FrictionSim2D aiida setup --use-remote
```

## Database and API setup (optional)

Initialize schema:

```bash
FrictionSim2D db init
```

Start the API server:

```bash
FrictionSim2D api serve --host 0.0.0.0 --port 8000
```

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
