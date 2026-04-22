# FrictionSim2D Documentation

Complete documentation for FrictionSim2D - A framework for 2D material friction simulations using LAMMPS.

## Documentation Structure

### Getting Started

1. **[Installation Guide](installation.md)** - Conda-based installation, prerequisites, and initial setup
2. **[Main Page](mainpage.md)** - Project overview, quick start, and architecture
3. **[Essentials](essentials.md)** - Core concepts and standard workflows

### Core Guides

4. **[Configuration Guide](configuration_guide.md)** 
   - Complete `.ini` file format reference
   - Parameter sweeps and material lists
   - AFM and sheet-on-sheet configurations
   - Advanced options and validation
   - Real-world examples with explanations

5. **[Python API Guide](python_api_guide.md)**
   - Programmatic simulation generation
   - High-level workflow functions
   - **Simplified AiiDA interface** (5 lines instead of 68!)
   - Configuration management
   - Batch processing and analysis
   - Error handling and best practices

6. **[AiiDA Workflows Guide](aiida_workflows.md)**
   - **New simplified interface** (v0.2.0)
   - Three execution modes (offline, local, remote)
   - Interactive prompting and auto-detection
   - CLI and Python examples
   - Migration guide from old interface
   - Troubleshooting guide

7. **[Settings Reference](settings_reference.md)** 
   - Complete `settings.yaml` documentation
   - All configuration sections explained
   - HPC cluster settings
   - Environment variables
   - Common patterns and best practices

### Reference

8. **[Commands Reference](commands.md)** - Complete CLI command documentation with examples
9. **[Examples](examples.md)** - End-to-end workflow examples

### Advanced Topics

10. **[HPC Two-Phase Jobs](HPC_TWO_PHASE_JOBS.md)** - AFM simulation job dependencies
11. **[Provenance Architecture](PROVENANCE_ARCHITECTURE.md)** - File tracking and reproducibility

## What's New in v0.2.0

### Simplified AiiDA Interface

**Before** (68 lines of boilerplate):
```python
from aiida.manage.configuration import load_profile
from aiida.orm import load_code, QueryBuilder
# ... 60+ more lines of setup, code detection, resource dict construction ...

processes = submit_batch(dirs, code, options={'resources': {...}})
```

**After** (5 lines):
```python
from src.aiida.submit import run_with_aiida

sims, root, nodes = run_with_aiida('config.ini', model='afm')
```

### Smart CLI Defaults

**Before**: Required 10+ parameters
```bash
FrictionSim2D aiida submit ./output --code lammps@hpc --machines 2 \\
  --mpiprocs 32 --walltime 12:00:00 --queue normal --project my_account
```

**After**: Auto-detect and prompt
```bash
FrictionSim2D aiida submit ./output
```

## Quick Navigation

### For New Users

Start here:
1. [Installation](installation.md) - Set up environment
2. [Main Page](mainpage.md) - Understand the tool
3. [Configuration Guide](configuration_guide.md) - Create your first config
4. [Examples](examples.md) - Run complete workflows

### For Python Users

Python-focused workflow:
1. [Python API Guide](python_api_guide.md) - Learn the API
2. [Configuration Guide](configuration_guide.md) - Config format
3. [Settings Reference](settings_reference.md) - Customize behavior
4. [AiiDA Workflows](aiida_workflows.md) - HPC submission

### For CLI Users

Command-line workflow:
1. [Commands Reference](commands.md) - All commands
2. [Configuration Guide](configuration_guide.md) - Config format
3. [AiiDA Workflows](aiida_workflows.md) - Submission strategies
4. [HPC Two-Phase Jobs](HPC_TWO_PHASE_JOBS.md) - HPC execution

### For HPC Users

Cluster-focused:
1. [Settings Reference](settings_reference.md) - Configure HPC settings
2. [AiiDA Workflows](aiida_workflows.md) - Submission modes
3. [HPC Two-Phase Jobs](HPC_TWO_PHASE_JOBS.md) - Job dependencies
4. [Commands Reference](commands.md) - HPC commands

## Common Tasks

### Generate Simulations

**CLI**:
```bash
FrictionSim2D run afm config.ini
```

**Python**:
```python
from src.core.run import run_simulations
sims, root = run_simulations('config.ini', model='afm')
```

See: [Commands Reference](commands.md) | [Python API Guide](python_api_guide.md)

### Submit to HPC with AiiDA

**CLI**:
```bash
FrictionSim2D aiida submit ./output
```

**Python**:
```python
from src.aiida.submit import run_with_aiida
sims, root, nodes = run_with_aiida('config.ini')
```

See: [AiiDA Workflows](aiida_workflows.md)

### Configure Settings

**View current**:
```bash
FrictionSim2D settings show
```

**Modify**: Edit `src/data/settings/settings.yaml`

See: [Settings Reference](settings_reference.md)

### Query Results

**CLI**:
```bash
FrictionSim2D aiida query --material MoS2 --format csv
```

**Python**:
```python
from aiida.orm import QueryBuilder, Dict
qb = QueryBuilder()
qb.append(Dict, filters={'attributes.material': 'MoS2'})
results = qb.all()
```

See: [AiiDA Workflows](aiida_workflows.md) | [Python API Guide](python_api_guide.md)

## Key Concepts

### Simulation Models

- **AFM**: Tip-substrate-sheet configuration for atomic force microscopy simulations
- **Sheet-on-sheet**: Bilayer sliding for direct friction studies

### Configuration Files

- `.ini` format with sections: `[general]`, `[2D]`, `[tip]`, `[sub]`
- Support parameter sweeps and material loops
- Validated with Pydantic models

### Provenance Tracking

- Automatic file manifest generation
- Component-level traceability (which potential for which part?)
- AiiDA integration for full workflow provenance

### HPC Execution

- Two-phase jobs for AFM (system.in → slide.in with dependencies)
- PBS and SLURM support
- Array job or batch submission modes

## Documentation Improvements (v0.2.0)

### New Documentation

- ✅ **Configuration Guide** (500+ lines) - Comprehensive config reference
- ✅ **Python API Guide** (600+ lines) - Complete API documentation
- ✅ **AiiDA Workflows Guide** (700+ lines) - Detailed AiiDA usage
- ✅ **Settings Reference** (600+ lines) - All settings explained

### Updated Documentation

- ✅ **Commands Reference** - Updated with new simplified interface
- ✅ **Examples** - (Next to update)
- ✅ **Installation** - (Existing, adequate)
- ✅ **Main Page** - (Next to update)
- ✅ **Essentials** - (Next to update)

### Total Addition

- **~2500 lines** of new comprehensive documentation
- **4 new major guides** covering all aspects
- **Updated examples** with new interface
- **Migration guides** from old to new API

## Support and Contribution

### Getting Help

1. Check relevant documentation section above
2. Review [Examples](examples.md) for similar use cases
3. Search [GitHub Issues](https://github.com/your-repo/FrictionSim2D/issues)
4. Open new issue with:
   - Command/code used
   - Config file (if applicable)
   - Error output
   - Environment info (`FrictionSim2D --version`, conda list)

### Contributing

Documentation improvements welcome! Please:
- Follow existing structure and style
- Include code examples
- Test examples before submitting
- Add cross-references to related sections

## Version History

### v0.2.0 (Current)
- ✨ New simplified AiiDA interface
- ✨ Smart CLI defaults and prompting
- ✨ Auto-detection for codes
- 📚 Comprehensive documentation overhaul
- 🐛 Two-phase job dependency fixes

### v0.1.0
- Initial release
- AFM and sheet-on-sheet models
- Basic AiiDA integration
- HPC script generation

## License

[Add your license information here]

## Citation

If FrictionSim2D is used in published work, please cite:
- FrictionSim2D repository and version
- LAMMPS (simulation engine)
- AiiDA (if using workflow features)

---

*Documentation last updated: 2026-02-12*
