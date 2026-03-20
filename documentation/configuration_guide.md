# Configuration File Guide

\anchor configuration_guide

## Overview

FrictionSim2D uses INI-format configuration files to define simulation parameters. These files support:
- **Parameter sweeps**: Multiple values for force, pressure, scan angle, and other general parameters
- **Material placeholders**: Use `{mat}` to loop over multiple materials
- **Path interpolation**: Automatically resolve potentials and CIF files
- **Model-specific sections**: AFM uses `[tip]`, `[sub]`, and `[2D]`; Sheet-on-sheet uses only `[2D]`

## Configuration File Structure

### AFM Configuration

AFM simulations require four sections: `[2D]`, `[tip]`, `[sub]`, and `[general]`.

AFM supports layer sweeps: multiple values in `[2D].layers` will generate a separate simulation for each layer count (e.g., layers=[1,2,3] creates L1, L2, and L3 subdirectories).

```ini
# 2D Material (sheet between tip and substrate)
[2D]
mat = h-MoS2
cif_path = structures/MoS2.cif
x = 100                    # Sheet length in nm
y = 100                    # Sheet width in nm
layers = [1,2,3]           # Number of layers (sweepable)
pot_path = MoS2.sw
pot_type = sw
stack_type = AB            # AB (Bernal) or AA stacking
lat_c = 6.2                # Interlayer spacing in Angstrom (optional)

# AFM Tip
[tip]
mat = Si
r = 25                     # Tip radius in Angstrom
amorph = c                 # 'c' = crystalline, 'a' = amorphous
dspring = 1.6e-6           # Damping constant
pot_path = Si.sw
pot_type = sw
cif_path = structures/Si.cif

# Substrate
[sub]
mat = Si
thickness = 12             # Substrate thickness in nm
amorph = a                 # 'a' = amorphous (quenched melt)
pot_path = Si.sw
pot_type = sw
cif_path = structures/Si.cif

# General simulation parameters
[general]
temp = 300                 # Temperature in Kelvin
force = [2,5,10,20]        # Normal forces in nN (sweepable)
scan_angle = 0             # Scan direction in degrees
scan_speed = 2             # Scan velocity in m/s
driving_spring = 50        # Spring constant for tip drive in N/m
```

### Sheet-on-Sheet Configuration

Sheet-on-sheet simulations use only `[2D]` and `[general]` sections.

Important constraints for this model:
- `layers` must contain exactly one value.
- The layer count must be at least 3 (bilayer is not supported in this model).
- Layer sweeps are not applied in this model builder.

```ini
# 2D Material
[2D]
mat = h-MoS2
cif_path = structures/MoS2.cif
x = 100                    # Sheet dimensions in nm
y = 100
layers = [3]               # Single value only, must be >= 3
pot_path = MoS2.sw
pot_type = sw
stack_type = AB

# General parameters
[general]
temp = 300
pressure = [-0.5, 0.1, 1, 10]    # Applied pressure in GPa (sweepable)
scan_angle = [0, 90, 15, 1]      # [start, end, step, force/pressure]
scan_speed = 1                    # Scan velocity in m/s
bond_spring = 5                   # Spring between sheets in eV/A^2
driving_spring = 50               # Drive spring in N/m
```

## Parameter Sweeps

### Single Parameter Sweeps

Use list notation for any sweepable parameter:

```ini
[general]
force = [5, 10, 20, 50]          # 4 simulations
# OR
pressure = [0.1, 1, 10]          # 3 simulations
```

### Multi-Parameter Sweeps

Combine multiple sweep parameters for Cartesian products:

```ini
[2D]
layers = [1, 2, 3]               # 3 values

[general]
force = [5, 10, 20]              # 3 values
# Total: 3 × 3 = 9 simulations
```

### Scan Angle Sweeps

Special 4-element format for scan angle ranges:

```ini
[general]
# Format: [start, end, step, force_or_pressure]
scan_angle = [0, 90, 15, 10]     # 0° to 90° every 15°, at force=10 nN
# Generates: 0°, 15°, 30°, 45°, 60°, 75°, 90° (7 simulations)
```

The 4th element selects which force/pressure value to use when combined with force/pressure sweeps.

### Material List Sweeps

Loop over multiple materials using a materials list file:

```ini
[2D]
materials_list = materials.txt   # Path to materials list
mat = {mat}                      # Placeholder replaced for each material
cif_path = structures/{mat}.cif  # Auto-interpolated paths
pot_path = potentials/{mat}.sw
```

**materials.txt**:
```
h-MoS2
h-WSe2
graphene
```

This generates one simulation per material.

## Material and Potential Paths

### Path Resolution Order

FrictionSim2D resolves paths in this order:

1. **Absolute paths**: `/home/user/potentials/MoS2.sw` (used as-is)
2. **Relative paths**: `../potentials/MoS2.sw` (relative to config file)
3. **Package data**: Searches `src/data/potentials/` and `src/data/materials/`
4. **Filename only**: `MoS2.sw` searches package data directories

### Potential Types

Supported potential formats:

| Type | Extension | Description |
|------|-----------|-------------|
| `sw` | `.sw` | Stillinger-Weber (3-body) |
| `tersoff` | `.tersoff` | Tersoff (3-body, covalent) |
| `eam` | `.eam`, `.eam.alloy` | Embedded Atom Method (metals) |
| `meam` | `.meam` | Modified EAM |
| `reax` | `.reax` | ReaxFF reactive force field |

### CIF Files

Crystal structure files must be in CIF format. FrictionSim2D uses Atomsk to:
- Read CIF structures
- Build supercells
- Generate LAMMPS data files
- Apply stacking patterns (AA vs AB)

## Advanced Configuration

### Custom LJ Overrides

You can override Lennard-Jones coefficients used for cross/interlayer interactions
by adding an optional `[lj_override]` section.

Each key is an element pair (order-independent), and each value is
`[epsilon, sigma]` in LAMMPS units.

```ini
[lj_override]
Mo-Mo = [1.0624, 3.878597]
Mo-S = [0.4124, 3.75114]
S-S = [0.198443, 3.62368]
```

Supported pair key formats include `Mo-S`, `Mo_S`, `Mo S`, and `Mo,S`.
If a pair is not listed, FrictionSim2D falls back to default UFF mixing rules.

### Amorphous Materials

Generate amorphous structures via quench-melt process:

```ini
[tip]
amorph = a                 # Enable amorphous generation

[sub]
amorph = a
```

Amorphous generation uses settings from `settings.yaml`:

```yaml
quench:
  melt_temp: 2500          # Melting temperature (K)
  quench_temp: 300         # Target temperature (K)
  quench_rate: 1e12        # Cooling rate (K/s)
  quench_slab_dims: [200, 200, 50]  # Quench cell size (Angstrom)
```

### Outer Loop Parameter

Control which parameter generates separate LAMMPS scripts:

```ini
[general]
pressure = [0.1, 1, 10]
scan_speed = [1, 2, 5]
outer_loop = pressure      # Creates slide_0.1.in, slide_1.in, slide_10.in
```

Without `outer_loop`, all combinations use a single `slide.in` script (legacy behavior).

### Custom Settings Override

Override default settings on a per-config basis:

```ini
[2D]
# ... material config ...

[general]
# ... simulation params ...

# Override settings
[settings_override]
simulation.timestep = 0.0005
simulation.slide_run_steps = 1000000
output.dump_frequency.slide = 5000
```

This merges with `settings.yaml`, with config values taking precedence.

## Validation

FrictionSim2D validates configurations using Pydantic models:

**Validation checks**:
- Required fields present
- Numeric values in valid ranges
- File paths resolve correctly
- Potential type matches file extension
- Layer count ≥ 1
- Temperature ≥ 0

**Common validation errors**:

```
ValueError: Tip radius must be positive
  → Check tip.r > 0

FileNotFoundError: CIF file not found: structures/MoS2.cif
  → Verify cif_path is correct and file exists

ValidationError: layers must be >= 1
  → Check [2D] layers parameter
```

## Configuration Examples

### Minimal AFM

```ini
[2D]
mat = graphene
cif_path = graphene.cif
x = 50
y = 50
layers = [1]
pot_path = C.sw
pot_type = sw

[tip]
mat = C
r = 20
pot_path = C.sw
pot_type = sw
cif_path = C.cif

[sub]
mat = C
thickness = 10
pot_path = C.sw
pot_type = sw
cif_path = C.cif

[general]
temp = 300
force = [10]
```

### Multi-Material Sheet-on-Sheet

```ini
[2D]
materials_list = materials.txt
mat = {mat}
cif_path = structures/{mat}.cif
x = 100
y = 100
layers = [3]
pot_path = potentials/{mat}.sw
pot_type = sw
stack_type = AB

[general]
temp = 300
pressure = [0.1, 1, 10]
scan_angle = [0, 90, 30, 1]
scan_speed = 1
```

### Complex AFM Sweep

```ini
[2D]
mat = MoS2
cif_path = MoS2.cif
x = 100
y = 100
layers = [1, 2, 3]         # 3 layer configs
pot_path = MoS2.sw
pot_type = sw

[tip]
mat = Si
r = 25
amorph = c
pot_path = Si.sw
pot_type = sw
cif_path = Si.cif

[sub]
mat = Si
thickness = 12
amorph = a                  # Amorphous substrate
pot_path = Si.sw
pot_type = sw
cif_path = Si.cif

[general]
temp = 300
force = [5, 10, 20, 50]    # 4 force values
scan_angle = [0, 90, 15, 10]  # 7 angles at F=10nN
scan_speed = 2
# Total: 3 layers × (4 forces + 7 angles) = 33 simulations
```

## Tips and Best Practices

1. **Start small**: Test with single values before expanding to sweeps
2. **Use materials_list**: Easier to manage multi-material studies
3. **Path conventions**: Keep potentials and CIFs in dedicated directories
4. **Validate early**: Run with `--dry-run` to check config parsing
5. **Amorphous caution**: Quenching is expensive; test locally first
6. **Outer loop strategy**: Use for long parameter lists to reduce array job size
7. **Document materials**: Keep notes on material IDs and sources
8. **Version control configs**: Track parameter changes over time

## See Also

- [Settings Reference](settings_reference.md) - `settings.yaml` options
- [Examples](examples.md) - Complete working examples
- [Python API Guide](python_api_guide.md) - Using configs in Python scripts
