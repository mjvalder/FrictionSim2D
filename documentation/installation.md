# Installation

This guide is aligned with the current repository (`pyproject.toml`) and CLI behavior.

## Requirements

- Python 3.9+
- LAMMPS executable available in your environment
- Atomsk executable available in your environment

Optional features require extra dependencies:

- AiiDA workflows: `aiida-core`
- Database commands: `psycopg2-binary`
- Plotting: `matplotlib`, `seaborn`, `scipy`
- API server: `fastapi`, `uvicorn`, `httpx`

## Install From Source (Recommended for this repository)

From the project root:

```bash
pip install -e .
```

Install optional extras as needed:

```bash
pip install -e .[aiida]
pip install -e .[db]
pip install -e .[plotting]
pip install -e .[api]
pip install -e .[all]
```

## Verify Installation

```bash
FrictionSim2D --help
FrictionSim2D --version
```

Then verify external binaries used by workflows:

```bash
lmp -h
atomsk --version
```

## AiiDA Setup (Optional)

After installing AiiDA dependencies:

```bash
FrictionSim2D aiida status
FrictionSim2D aiida setup
```

If your workflows target a remote machine, configure the `aiida` section in settings and use:

```bash
FrictionSim2D aiida setup --use-remote
```

## Database/API Setup (Optional)

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

`lmp` or `atomsk` missing

- Install these tools in the same environment used to run FrictionSim2D.

AiiDA commands abort immediately

- Install AiiDA dependencies (`pip install -e .[aiida]`), then rerun `FrictionSim2D aiida setup`.
