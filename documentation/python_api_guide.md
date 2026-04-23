# Python API Guide

This guide focuses on package APIs exposed in `src/__init__.py` and core modules.

## High-Level Imports

```python
from src import afm, sheetonsheet, run_simulations, load_settings, parse_config
```

Available builders:

```python
from src import AFMSimulation, SheetOnSheetSimulation
```

If AiiDA extras are installed, `run_with_aiida` is also exported.

## 1) Run AFM from Python

```python
from src import afm

afm(config_file="afm_config.ini", output_root="./simulation_output", generate_hpc=True)
```

## 2) Run Sheet-on-Sheet from Python

```python
from src import sheetonsheet

sheetonsheet(config_file="sheet_config.ini", output_root="./simulation_output")
```

## 3) Use `run_simulations` Directly

```python
from src.core.run import run_simulations

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

## 4) Parse and Validate Configs

```python
from src.core.config import parse_config, AFMSimulationConfig, load_settings

raw = parse_config("afm_config.ini")
raw["settings"] = load_settings().model_dump()
validated = AFMSimulationConfig(**raw)
```

## 5) Builder-Level Usage

```python
from src.builders.afm import AFMSimulation
from src.core.config import AFMSimulationConfig, parse_config, load_settings

cfg = parse_config("afm_config.ini")
cfg["settings"] = load_settings().model_dump()
obj = AFMSimulationConfig(**cfg)

sim = AFMSimulation(config=obj, output_dir="./single_case")
sim.set_base_output_dir("./")
sim.build()
```

## 6) AiiDA Helper

```python
from src.aiida.submit import run_with_aiida

sims, root, nodes = run_with_aiida("afm_config.ini", model="afm")
```

Use this for generate+submit flow with one call.

## Notes

- `src.__init__` defines convenience wrappers `afm()` and `sheetonsheet()`.
- Internal helper `_run_all()` exists in source but is intentionally internal by naming convention.
- For long sweeps, call patterns that return created paths and log metadata immediately to your run notebook or database.
