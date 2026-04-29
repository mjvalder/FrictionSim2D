"""AiiDA CalcJob and Parser for LAMMPS friction simulations.

Provides :class:`LammpsFrictionCalcJob`, which submits pre-built
FrictionSim2D simulation directories to an HPC cluster via AiiDA's
job scheduling infrastructure, and :class:`LammpsFrictionParser`,
which extracts results from the retrieved outputs.
"""

from __future__ import annotations

import io
import logging
import re
import shlex
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from aiida import orm
from aiida.common.datastructures import CalcInfo, CodeInfo
from aiida.common.folders import Folder
from aiida.engine import CalcJob
from aiida.parsers.parser import Parser

logger = logging.getLogger(__name__)

_SIM_SUBDIRS = (
    'lammps', 'data', 'potentials', 'results', 'visuals', 'build', 'provenance',
)


# ---------------------------------------------------------------------------
# CalcJob
# ---------------------------------------------------------------------------

class LammpsFrictionCalcJob(CalcJob):
    """AiiDA CalcJob that runs a FrictionSim2D LAMMPS simulation."""

    _DEFAULT_LAMMPS_SCRIPTS = ['system.in', 'slide.in']

    @classmethod
    def define(cls, spec):
        """Define the CalcJob input/output specification."""
        super().define(spec)

        spec.input('code', valid_type=orm.AbstractCode,
                   help='LAMMPS executable code node.')
        spec.input('simulation_dir', valid_type=orm.FolderData,
                   help='FolderData with the simulation directory contents.')
        spec.input('provenance', valid_type=orm.Data, required=False,
                   help='FrictionProvenanceData node for provenance tracking.')
        spec.input('simulation_node', valid_type=orm.Data, required=False,
                   help='FrictionSimulationData node for status tracking.')
        spec.input('parameters', valid_type=orm.Dict, required=False,
                   default=lambda: orm.Dict({}),
                   help='Optional runtime parameters.')

        spec.input('metadata.options.resources', valid_type=dict,
                   default={'num_machines': 1, 'num_mpiprocs_per_machine': 32})
        spec.input('metadata.options.max_wallclock_seconds', valid_type=int,
                   default=72000)
        spec.input('metadata.options.parser_name', valid_type=str,
                   default='friction2d.lammps')

        spec.output('results_folder', valid_type=orm.FolderData, required=False,
                    help='Retrieved results directory.')
        spec.output('log_file', valid_type=orm.SinglefileData, required=False,
                    help='LAMMPS log file.')
        spec.output('stdout_folder', valid_type=orm.FolderData, required=False,
                    help='FolderData with LAMMPS stdout files (*.out).')

        spec.exit_code(300, 'ERROR_NO_LAMMPS_SCRIPTS',
                       message='No LAMMPS input scripts found in simulation directory.')
        spec.exit_code(310, 'ERROR_LAMMPS_FAILED',
                       message='LAMMPS returned a non-zero exit code.')
        spec.exit_code(320, 'ERROR_NO_RESULTS',
                       message='No results files were produced.')

    def prepare_for_submission(self, folder: Folder) -> CalcInfo:
        """Prepare the calculation for submission."""
        inputs = self.inputs  # type: ignore[attr-defined]
        sim_folder = inputs.simulation_dir
        repo = sim_folder.base.repository

        params = (inputs.get('parameters', None) or orm.Dict({})).get_dict()
        scripts_override = params.get('lammps_scripts')
        lammps_flags = params.get('lammps_flags', '-l none')
        lammps_flag_list = (
            shlex.split(lammps_flags) if isinstance(lammps_flags, str)
            else list(lammps_flags)
        )

        lammps_scripts = (
            _collect_lammps_scripts(sim_folder, scripts_override)
            or self._DEFAULT_LAMMPS_SCRIPTS
        )
        sim_prefix = _detect_sim_prefix(repo, lammps_scripts)

        _stage_files_into_sandbox(folder, repo, sim_prefix)
        _handle_array_mode(folder, sim_folder, params, scripts_override, lammps_scripts)

        code_infos = [
            _make_code_info(inputs.code.uuid, lammps_flag_list, script)
            for script in lammps_scripts
        ]

        calcinfo = CalcInfo()
        calcinfo.uuid = self.uuid
        calcinfo.codes_info = code_infos
        calcinfo.retrieve_list = [
            'results/*', 'visuals/*', 'log.lammps', '*.out', 'array_status/*',
        ]
        if sim_prefix:
            calcinfo.retrieve_list.extend([
                f'{sim_prefix}/results/*',
                f'{sim_prefix}/visuals/*',
            ])
        return calcinfo


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class LammpsFrictionParser(Parser):
    """Parse retrieved outputs for FrictionSim2D LAMMPS runs."""

    def parse(self, **kwargs):  # type: ignore[override]
        """Parse retrieved files and create output nodes."""
        retrieved = self.retrieved
        if retrieved is None:
            return self.exit_codes.ERROR_NO_RESULTS

        repo = retrieved.base.repository
        names = repo.list_object_names()

        result_names = [n for n in names if _is_result_file(n)]
        visual_names = [n for n in names if _is_visual_file(n)]
        stdout_names = [n for n in names if n.endswith('.out')]
        has_log = 'log.lammps' in names

        if not result_names and not visual_names and not stdout_names and not has_log:
            return self.exit_codes.ERROR_NO_RESULTS

        self._store_results(repo, result_names + visual_names)
        self._store_log(repo, has_log)
        self._store_stdout(repo, stdout_names)
        self._copy_local(repo, names)

        if not result_names and not visual_names:
            return self.exit_codes.ERROR_LAMMPS_FAILED
        return None

    def _store_results(self, repo, file_names):
        """Store results and visuals as FolderData."""
        if not file_names:
            return
        results_folder = orm.FolderData()
        for name in file_names:
            with repo.open(name, 'rb') as handle:
                results_folder.base.repository.put_object_from_filelike(handle, name)
        results_folder.store()
        self.out('results_folder', results_folder)

    def _store_log(self, repo, has_log):
        """Store LAMMPS log file as SinglefileData."""
        if not has_log:
            return
        with repo.open('log.lammps', 'rb') as handle:
            with NamedTemporaryFile(prefix='lammps_log_', suffix='.log',
                                    delete=True) as tmp:
                tmp.write(handle.read())
                tmp.flush()
                log_file = orm.SinglefileData(file=tmp.name)
        log_file.store()
        self.out('log_file', log_file)

    def _store_stdout(self, repo, stdout_names):
        """Store stdout files as FolderData."""
        if not stdout_names:
            return
        stdout_folder = orm.FolderData()
        for name in stdout_names:
            with repo.open(name, 'rb') as handle:
                stdout_folder.base.repository.put_object_from_filelike(handle, name)
        stdout_folder.store()
        self.out('stdout_folder', stdout_folder)

    def _copy_local(self, repo, names):
        """Copy outputs to local simulation directory if configured."""
        params = getattr(self.node.inputs, 'parameters', None)
        if params is None:
            return
        local_sim_dir = params.get_dict().get('local_sim_dir')
        if not local_sim_dir:
            return
        _copy_outputs_to_local(repo, names, Path(local_sim_dir))


# ---------------------------------------------------------------------------
# Preparation helpers (public API)
# ---------------------------------------------------------------------------

def prepare_simulation_folder(sim_dir: Path) -> orm.FolderData:
    """Create an AiiDA ``FolderData`` from a simulation directory.

    Args:
        sim_dir: Path to the simulation directory (must contain ``lammps/``).

    Returns:
        A new (unstored) ``FolderData`` node.
    """
    sim_dir = Path(sim_dir)
    if not (sim_dir / 'lammps').exists():
        raise FileNotFoundError(f"No lammps/ directory in {sim_dir}")

    folder_data = orm.FolderData()
    _stage_sim_dir(folder_data, sim_dir, relative_to=sim_dir)
    return folder_data


def prepare_simulation_root(root_dir: Path, simulation_dirs: list) -> orm.FolderData:
    """Create a FolderData containing multiple simulation directories.

    Args:
        root_dir: Root directory containing simulation subdirectories.
        simulation_dirs: List of simulation directories under root_dir.

    Returns:
        A new (unstored) ``FolderData`` node with staged inputs.
    """
    root_dir = Path(root_dir)
    folder_data = orm.FolderData()
    for sim_dir in simulation_dirs:
        _stage_sim_dir(folder_data, Path(sim_dir), relative_to=root_dir)
    return folder_data


def apply_options(builder, options: dict[str, Any] | None) -> None:
    """Apply metadata options to a CalcJob builder."""
    if not options:
        return
    for key in ('resources', 'max_wallclock_seconds', 'queue_name',
                'account', 'custom_scheduler_commands', 'prepend_text'):
        if key in options:
            setattr(builder.metadata.options, key, options[key])


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _is_result_file(name: str) -> bool:
    """Check whether a retrieved file belongs to results."""
    return (name.startswith('results/') or '/results/' in name
            or name.startswith('friction_'))


def _is_visual_file(name: str) -> bool:
    """Check whether a retrieved file belongs to visuals."""
    return (name.startswith('visuals/') or '/visuals/' in name
            or name.endswith('.lammpstrj'))


def _stage_sim_dir(folder_data: orm.FolderData, sim_dir: Path,
                   relative_to: Path) -> None:
    """Stage simulation subdirectories into a FolderData node."""
    for subdir in _SIM_SUBDIRS:
        sub_path = sim_dir / subdir
        if sub_path.exists():
            for file_path in sub_path.rglob('*'):
                if file_path.is_file():
                    rel = file_path.relative_to(relative_to)
                    folder_data.base.repository.put_object_from_file(
                        str(file_path), str(rel),
                    )


def _stage_files_into_sandbox(folder: Folder, repo, sim_prefix: str | None) -> None:
    """Copy all repository files into the AiiDA sandbox folder."""
    for subdir in _SIM_SUBDIRS:
        folder.get_subfolder(subdir, create=True)
        if sim_prefix:
            folder.get_subfolder(f"{sim_prefix}/{subdir}", create=True)

    for root, _dirs, files in repo.walk():
        for file_name in files:
            rel_path = (root / file_name).as_posix()
            with repo.open(rel_path, 'rb') as src:
                payload = src.read()

            if root.as_posix() == '.':
                folder.create_file_from_filelike(io.BytesIO(payload), file_name)
            else:
                sub = folder.get_subfolder(root.as_posix(), create=True)
                sub.create_file_from_filelike(io.BytesIO(payload), file_name)

            if sim_prefix:
                pref_root = (Path(sim_prefix) / root).as_posix()
                pref = folder.get_subfolder(pref_root, create=True)
                pref.create_file_from_filelike(io.BytesIO(payload), file_name)


def _handle_array_mode(folder, sim_folder, params, scripts_override,
                       lammps_scripts) -> None:
    """Write array-mode helper files into the sandbox if needed."""
    if not params.get('array_mode'):
        return

    if 'array_map.txt' not in sim_folder.base.repository.list_object_names():
        raise ValueError("array_map.txt missing for array mode")

    if scripts_override:
        scripts_payload = "\n".join(lammps_scripts).encode('utf-8')
        folder.create_file_from_filelike(
            io.BytesIO(scripts_payload), 'array_scripts.txt',
        )

    wrapper = _build_array_wrapper(
        launcher_cmd=params.get('launcher_cmd', 'mpirun'),
        lammps_exec=params.get('lammps_executable', 'lmp'),
        lammps_flags=params.get('lammps_flags', '-l none'),
    )
    folder.create_file_from_filelike(
        io.BytesIO(wrapper.encode('utf-8')), 'run_array.sh',
    )


def _make_code_info(code_uuid: str, flag_list: list, script: str) -> CodeInfo:
    """Build a CodeInfo for a single LAMMPS script."""
    codeinfo = CodeInfo()
    codeinfo.code_uuid = code_uuid
    codeinfo.cmdline_params = flag_list + ['-in', f'lammps/{script}']
    codeinfo.stdout_name = f'{script}.out'
    codeinfo.withmpi = True
    return codeinfo


def _collect_lammps_scripts(sim_folder, scripts_override):
    """Return ordered LAMMPS scripts, or override list if provided."""
    if scripts_override:
        if isinstance(scripts_override, str):
            return [scripts_override]
        return [str(script) for script in scripts_override]

    repo = sim_folder.base.repository
    discovered = [
        name for name in repo.list_object_names('lammps')
        if name.endswith('.in')
    ]
    system = [s for s in discovered if s.lower() == 'system.in']
    slides = sorted(s for s in discovered
                    if s.lower() != 'system.in' and 'slide' in s.lower())
    others = sorted(s for s in discovered
                    if s.lower() != 'system.in' and 'slide' not in s.lower())
    return system + slides + others

def _detect_sim_prefix(repo, lammps_scripts):
    """Infer the simulation path prefix from LAMMPS scripts, if any."""
    pattern = re.compile(r"\bread_data\s+(\S+)")
    for script in lammps_scripts:
        path = f"lammps/{script}"
        try:
            with repo.open(path, 'r') as handle:
                for line in handle:
                    match = pattern.search(line)
                    if match and '/build/' in match.group(1):
                        return match.group(1).split('/build/', 1)[0]
        except FileNotFoundError:
            continue
    return None


def _build_array_wrapper(launcher_cmd: str, lammps_exec: str,
                         lammps_flags: str) -> str:
    """Build a TMPDIR staging wrapper for array jobs."""
    return f"""#!/usr/bin/env bash
set -euo pipefail

IDX="${{PBS_ARRAY_INDEX:-${{SLURM_ARRAY_TASK_ID:-1}}}}"

mapfile -t SIMS < array_map.txt
SIM_REL="${{SIMS[$((IDX-1))]:-}}"
if [[ -z "${{SIM_REL}}" ]]; then
    echo "No simulation for index ${{IDX}}" >&2
    exit 2
fi

SIM_ROOT="$PWD"
RUN_ROOT="${{TMPDIR:-/tmp}}/friction2d_${{IDX}}"
LMP_FLAGS="{lammps_flags}"

mkdir -p "$RUN_ROOT"
mkdir -p "$SIM_ROOT/array_status"

trap 'echo "{{\\\\\"index\\\\\": $IDX, \\\\\"simulation\\\\\": \\\\\"$SIM_REL\\\\\", \\\\\"status\\\\\": \\\\\"failed\\\\\"}}\" \\
    > "$SIM_ROOT/array_status/${{IDX}}.json"' ERR

echo "{{\\\\\"index\\\\\": $IDX, \\\\\"simulation\\\\\": \\\\\"$SIM_REL\\\\\", \\\\\"status\\\\\": \\\\\"running\\\\\"}}\" \\
    > "$SIM_ROOT/array_status/${{IDX}}.json"

rsync -a --exclude='*.lammpstrj' "$SIM_ROOT/$SIM_REL/" "$RUN_ROOT/$SIM_REL/"

cd "$RUN_ROOT/$SIM_REL"

if [[ -f "$SIM_ROOT/array_scripts.txt" ]]; then
    mapfile -t SCRIPTS < "$SIM_ROOT/array_scripts.txt"
else
    mapfile -t ALLSCRIPTS < <(find lammps -maxdepth 1 -type f -name '*.in' -printf '%f\\\\n' | sort)
    SYSTEM=()
    SLIDES=()
    OTHERS=()
    for s in "${{ALLSCRIPTS[@]}}"; do
        low="${{s,,}}"
        if [[ "$low" == "system.in" ]]; then
            SYSTEM+=("$s")
        elif [[ "$low" == *slide* ]]; then
            SLIDES+=("$s")
        else
            OTHERS+=("$s")
        fi
    done
    SCRIPTS=("${{SYSTEM[@]}}" "${{SLIDES[@]}}" "${{OTHERS[@]}}")
fi

for script in "${{SCRIPTS[@]}}"; do
    {launcher_cmd} {lammps_exec} $LMP_FLAGS -in "lammps/$script"
done

for sub in results visuals; do
    if [[ -d "$RUN_ROOT/$SIM_REL/$sub" ]]; then
        rsync -a "$RUN_ROOT/$SIM_REL/$sub" "$SIM_ROOT/$SIM_REL/"
    fi
done

echo "{{\\\\\"index\\\\\": $IDX, \\\\\"simulation\\\\\": \\\\\"$SIM_REL\\\\\", \\\\\"status\\\\\": \\\\\"completed\\\\\"}}\" \\
    > "$SIM_ROOT/array_status/${{IDX}}.json"
"""


def _copy_outputs_to_local(repo, names, local_sim_dir: Path) -> None:
    """Copy results and visuals into the local simulation directory."""
    results_dir = local_sim_dir / 'results'
    visuals_dir = local_sim_dir / 'visuals'
    results_dir.mkdir(parents=True, exist_ok=True)
    visuals_dir.mkdir(parents=True, exist_ok=True)

    for name in names:
        kind, rel_name = _categorize_output(name)
        if kind is None:
            continue
        target_dir = results_dir if kind == 'results' else visuals_dir
        target_path = target_dir / rel_name
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with repo.open(name, 'rb') as handle:
            target_path.write_bytes(handle.read())


def _categorize_output(name: str) -> tuple[str | None, str]:
    """Categorize a retrieved file as 'results', 'visuals', or None."""
    for kind, sep in (('results', '/results/'), ('visuals', '/visuals/')):
        if sep in name:
            return kind, name.split(sep, 1)[1]
        if name.startswith(f'{kind}/'):
            return kind, name.split(f'{kind}/', 1)[1]
    if name.startswith('friction_'):
        return 'results', name
    if name.endswith('.lammpstrj'):
        return 'visuals', name
    return None, name
