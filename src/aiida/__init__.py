"""AiiDA plugin for FrictionSim2D.

This module provides integration with AiiDA for managing friction simulations,
including:

- Custom data types for storing simulation metadata and results
- HPC script generation for PBS and SLURM schedulers
- Job manifest system for offline HPC workflows
- Query interface (Friction2DDB) for accessing stored data
- CLI commands for workflow management
- Workflow orchestration for the complete simulation pipeline

Typical Usage (Offline HPC Workflow):
--------------------------------------

1. **Prepare simulations**::

    from src.aiida.workflows import FrictionSimWorkflow
    
    workflow = FrictionSimWorkflow(
        config_path='afm_config.ini',
        output_dir='./friction_output'
    )
    workflow.prepare()

2. **Export for HPC**::

    workflow.export(scheduler='pbs')
    # Creates ./friction_output_hpc/ ready for transfer

3. **Manual HPC execution**:
   - Transfer package to HPC
   - Run: cd scripts && ./submit_all.sh
   - Transfer results back

4. **Import and postprocess**::

    workflow.import_results('./returned_results')
    workflow.postprocess()

5. **Query results**::

    from src.aiida.db import Friction2DDB
    
    db = Friction2DDB()
    results = db.query_by_material('h-MoS2')
    df = results.to_dataframe()

CLI Usage:
----------
    # Generate simulations
    friction2d prepare afm_config.ini -o ./output
    
    # Export for HPC
    friction2d export ./output -o ./hpc_package -s pbs
    
    # Import results
    friction2d import ./returned_results
    
    # Query database
    friction2d query -m h-MoS2 -l 2

"""

# Data types
from .data import (
    FrictionSimulationData,
    FrictionConfigData,
    FrictionResultsData,
    FrictionProvenanceData,
)

# HPC utilities
from .hpc import (
    HPCScriptGenerator,
    HPCConfig,
    JobManifest,
    JobEntry,
    JobStatus,
)

# Database interface
from .db import Friction2DDB

# Workflows
from .workflows import (
    FrictionSimWorkflow,
    PreparationWorkflow,
    PostProcessWorkflow,
)

# CLI
from .cli.commands import friction2d as cli

__all__ = [
    # Data types
    'FrictionSimulationData',
    'FrictionConfigData',
    'FrictionResultsData',
    'FrictionProvenanceData',
    # HPC
    'HPCScriptGenerator',
    'HPCConfig',
    'JobManifest',
    'JobEntry',
    'JobStatus',
    # Database
    'Friction2DDB',
    # Workflows
    'FrictionSimWorkflow',
    'PreparationWorkflow',
    'PostProcessWorkflow',
    # CLI
    'cli',
]
