# Python API

Driving the package from Python instead of the CLI, using the APIs exposed in
`frictionsim2d/__init__.py` and the package modules.

> `frictionsim2d/core/` was split into `core/` + `run/` + `potentials/` + `lattice/`. The old
> `frictionsim2d.core.run` / `frictionsim2d.core.potential_manager` paths no longer exist — import from
> the canonical modules shown below. See CLAUDE.md for the full mapping.

## High-level imports

```python
from frictionsim2d import afm, sheetonsheet, run_simulations, load_settings, parse_config
```

Available builders:

```python
from frictionsim2d import AFMSimulation, SheetOnSheetSimulation
```


## 1. Run AFM from Python

```python
from frictionsim2d import afm

afm(config_file="afm_config.ini", output_root="./simulation_output", generate_hpc=True)
```

## 2. Run sheet-on-sheet from Python

```python
from frictionsim2d import sheetonsheet

sheetonsheet(config_file="sheet_config.ini", output_root="./simulation_output")
```

## 3. Use `run_simulations` directly

```python
from frictionsim2d.run.simulate import run_simulations

created, root, expanded, settings = run_simulations(
    config_file="afm_config.ini",
    model="afm",
    output_root="./simulation_output",
    generate_hpc=False,
)

print(len(created), root)
```

Return values:

- `created`: list of created simulation directories
- `root`: simulation root directory
- `expanded`: expanded config dictionaries after sweep processing
- `settings`: loaded global settings object

## 4. Parse and validate configs

```python
from frictionsim2d.core.config import parse_config, AFMSimulationConfig, load_settings

raw = parse_config("afm_config.ini")
raw["settings"] = load_settings().model_dump()
validated = AFMSimulationConfig(**raw)
```

## 5. Builder-level usage

```python
from frictionsim2d.builders.afm import AFMSimulation
from frictionsim2d.core.config import AFMSimulationConfig, parse_config, load_settings

cfg = parse_config("afm_config.ini")
cfg["settings"] = load_settings().model_dump()
obj = AFMSimulationConfig(**cfg)

sim = AFMSimulation(config=obj, output_dir="./single_case")
sim.set_base_output_dir("./")
sim.build()
```

## 6. AiiDA store

AiiDA records finished runs; it never runs them (there is no submission
helper). With the AiiDA extras installed:

```python
from frictionsim2d.aiida import load_aiida_profile
from frictionsim2d.aiida.ingest import import_root
from frictionsim2d.aiida.query import Friction2DDB, point_table, related

load_aiida_profile()                      # settings.aiida.profile, else AiiDA's default
report = import_root("simulation_20260421_143404", label="260421-afm")
print(report.render())

db = Friction2DDB()
df = db.query(set_label="260421-afm").to_dataframe()   # one row per live point

table = point_table()
same_but_load = related(df.point_uuid[0], "load", table=table)
```

See [aiida.md](aiida.md) for the graph, `related()` and the archive export.

## Notes

- `frictionsim2d.__init__` defines convenience wrappers `afm()` and `sheetonsheet()`.
- Internal helper `_run_all()` exists in source but is intentionally internal by naming convention.
- For long sweeps, call patterns that return created paths and log metadata immediately to your run notebook.

## Related docs

- [configuration.md](configuration.md)
- [settings.md](settings.md)
- [aiida.md](aiida.md)
