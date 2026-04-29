"""AiiDA WorkChain for end-to-end friction simulation workflows.

Provides :class:`FrictionWorkChain`, which orchestrates the complete
lifecycle of a friction simulation on HPC:

1. **Generate** — Build simulation files from config.
2. **Submit** — Launch LAMMPS CalcJobs for each simulation.
3. **Monitor** — Track completion via the AiiDA daemon.
4. **Import** — Parse results and create ``FrictionResultsData`` nodes.

This enables fully automated workflows where ``FrictionSim2D run afm
config.ini --aiida`` generates files, submits to HPC, and imports
results—all tracked with AiiDA provenance.
"""

import logging
from pathlib import Path
from typing import Any, cast

from aiida import orm
from aiida.engine import ToContext
from aiida.engine.processes.workchains.workchain import WorkChain

from .calcjob import LammpsFrictionCalcJob, prepare_simulation_folder

logger = logging.getLogger(__name__)

_WORKFLOW_EXCEPTIONS = (OSError, ValueError, KeyError, TypeError, RuntimeError)


class FrictionWorkChain(WorkChain):
    """WorkChain for end-to-end friction simulation management.

    Orchestrates file generation, LAMMPS submission, and result import
    as a single AiiDA workflow with full provenance.

    Inputs:
        code: LAMMPS executable ``Code`` node.
        config_json: ``Dict`` with the parsed FrictionSim2D config.
        simulation_dirs: ``List`` of paths to pre-built simulation directories.
        options: ``Dict`` with HPC resource options.

    Outputs:
        simulation_nodes: ``List`` of ``FrictionSimulationData`` UUIDs.
        result_nodes: ``List`` of ``FrictionResultsData`` UUIDs.
    """

    @classmethod
    def define(cls, spec):
        """Define the WorkChain specification."""
        super_cls = cast(Any, super())
        super_cls.define(spec)

        # Inputs
        spec.input(
            'code',
            valid_type=orm.AbstractCode,
            help='LAMMPS executable code.',
        )
        spec.input(
            'simulation_dirs',
            valid_type=orm.List,
            help='List of simulation directory paths (absolute).',
        )
        spec.input(
            'config_path',
            valid_type=orm.Str,
            required=False,
            help='Path to original config file for registration.',
        )
        spec.input(
            'options',
            valid_type=orm.Dict,
            required=False,
            default=lambda: orm.Dict({
                'resources': {'num_machines': 1, 'num_mpiprocs_per_machine': 32},
                'max_wallclock_seconds': 72000,
            }),
            help='HPC resource options for each CalcJob.',
        )

        # Outputs
        spec.output(
            'simulation_uuids',
            valid_type=orm.List,
            required=False,
            help='UUIDs of registered FrictionSimulationData nodes.',
        )
        spec.output(
            'result_uuids',
            valid_type=orm.List,
            required=False,
            help='UUIDs of imported FrictionResultsData nodes.',
        )

        # Outline
        spec.outline(
            cls.register_simulations,
            cls.submit_calculations,
            cls.inspect_calculations,
            cls.import_results,
        )

        # Exit codes
        spec.exit_code(400, 'ERROR_NO_SIMULATIONS',
                        message='No simulation directories provided.')
        spec.exit_code(410, 'ERROR_ALL_SUBMISSIONS_FAILED',
                        message='All CalcJob submissions failed.')
        spec.exit_code(420, 'ERROR_SOME_CALCULATIONS_FAILED',
                        message='Some calculations failed (partial results available).')

    def register_simulations(self):
        """Step 1: Register simulation directories with AiiDA."""
        inputs = self.inputs  # type: ignore[attr-defined]
        ctx = self.ctx  # type: ignore[attr-defined]
        exit_codes = self.exit_codes  # type: ignore[attr-defined]

        sim_dirs = inputs.simulation_dirs.get_list()
        if not sim_dirs:
            return exit_codes.ERROR_NO_SIMULATIONS

        config_path = None
        if 'config_path' in inputs:
            config_path = Path(inputs.config_path.value)

        from .integration import register_single_simulation  # pylint: disable=import-outside-toplevel

        uuids = []
        ctx.sim_nodes = {}

        for sim_path_str in sim_dirs:
            sim_dir = Path(sim_path_str)
            if not sim_dir.exists():
                logger.warning("Simulation directory does not exist: %s", sim_dir)
                continue

            uuid = None
            if config_path:
                uuid = register_single_simulation(sim_dir, config_path)

            if uuid:
                uuids.append(uuid)
                ctx.sim_nodes[sim_path_str] = orm.load_node(uuid)

        self.out('simulation_uuids', orm.List(list=uuids).store())  # type: ignore[attr-defined]
        ctx.sim_dirs = sim_dirs
        logger.info("Registered %d / %d simulations", len(uuids), len(sim_dirs))
        return None

    def submit_calculations(self):
        """Step 2: Submit CalcJobs for each simulation."""
        inputs = self.inputs  # type: ignore[attr-defined]
        ctx = self.ctx  # type: ignore[attr-defined]
        exit_codes = self.exit_codes  # type: ignore[attr-defined]

        options = inputs.options.get_dict()
        calcs = {}

        for sim_path_str in ctx.sim_dirs:
            sim_dir = Path(sim_path_str)
            if not (sim_dir / 'lammps').exists():
                logger.warning("No lammps/ directory in %s — skipping", sim_dir)
                continue

            try:
                folder = prepare_simulation_folder(sim_dir)
                folder.store()

                builder = LammpsFrictionCalcJob.get_builder()  # type: ignore[attr-defined]
                builder = cast(Any, builder)
                builder.code = inputs.code
                builder.simulation_dir = folder
                builder.parameters = orm.Dict({'local_sim_dir': str(sim_dir)})

                if sim_path_str in ctx.sim_nodes:
                    builder.simulation_node = ctx.sim_nodes[sim_path_str]

                metadata = getattr(builder, 'metadata')
                if 'resources' in options:
                    metadata.options.resources = options['resources']
                if 'max_wallclock_seconds' in options:
                    metadata.options.max_wallclock_seconds = options['max_wallclock_seconds']

                calc_label = sim_dir.name
                calcs[calc_label] = self.submit(builder)  # type: ignore[attr-defined]
                logger.info("Submitted CalcJob for %s", calc_label)

            except _WORKFLOW_EXCEPTIONS:
                logger.warning("Failed to submit %s", sim_dir, exc_info=True)

        if not calcs:
            return exit_codes.ERROR_ALL_SUBMISSIONS_FAILED

        return ToContext(**calcs)

    def inspect_calculations(self):
        """Step 3: Check CalcJob results and update simulation node statuses."""
        ctx = self.ctx  # type: ignore[attr-defined]
        exit_codes = self.exit_codes  # type: ignore[attr-defined]

        n_ok = 0
        n_fail = 0
        ctx.completed_dirs = []

        for label, calc in ctx.items():
            if not hasattr(calc, 'is_finished'):
                continue

            # Find corresponding simulation node
            sim_node = ctx.sim_nodes.get(label)

            if calc.is_finished_ok:
                n_ok += 1
                if sim_node:
                    sim_node.status = 'completed'
                ctx.completed_dirs.append(label)
            else:
                n_fail += 1
                if sim_node:
                    sim_node.status = 'failed'
                logger.warning(
                    "CalcJob %s failed with exit status %s",
                    label, calc.exit_status,
                )

        logger.info("Calculations: %d succeeded, %d failed", n_ok, n_fail)

        if n_ok == 0 and n_fail > 0:
            return exit_codes.ERROR_ALL_SUBMISSIONS_FAILED
        if n_fail > 0:
            self.report(f"WARNING: {n_fail} calculations failed")  # type: ignore[attr-defined]
        return None

    def import_results(self):
        """Step 4: Import results from completed calculations.

        Creates ``FrictionResultsData`` nodes and links them to the
        corresponding ``FrictionSimulationData`` nodes.
        """
        ctx = self.ctx  # type: ignore[attr-defined]
        from .integration import import_results_to_aiida  # pylint: disable=import-outside-toplevel

        result_uuids = []
        for label in ctx.completed_dirs:
            calc = getattr(ctx, label, None)
            if calc is None:
                continue

            if 'results_folder' in calc.outputs:
                try:
                    # The results are in the retrieved FolderData
                    uuids = import_results_to_aiida(
                        Path(calc.outputs.results_folder.get_remote_path())
                    )
                    result_uuids.extend(uuids)
                except _WORKFLOW_EXCEPTIONS:
                    logger.warning(
                        "Failed to import results for %s", label, exc_info=True
                    )

        if result_uuids:
            self.out(  # type: ignore[attr-defined]
                'result_uuids',
                orm.List(list=result_uuids).store(),
            )

        logger.info("Imported %d result nodes", len(result_uuids))
