"""Subprocess entry point for a preview build (see :mod:`.worker_common`).

Construction calls LAMMPS in-process: no cancel point in a thread, and a
C-level abort would take the window down.
"""
# pylint: disable=import-outside-toplevel  # the build stack loads only once a request is read

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

from .worker_common import emit as _emit, run_worker


def run_quench(request: Dict[str, Any]) -> Dict[str, Any]:
    """Preview a quench config: the crystal slab of its first member.

    Nothing is melted. The member's root is written to the preview directory
    (a crystal block and a deck, which is what Generate would write) and only
    the block is read back.
    """
    from ..builders.construction.amorphous import (
        member_rel_dir, read_data_header, write_quench_root)
    from ..core.config import load_settings, parse_config
    from ..core.utils.structures import cifread
    from .controllers.quench import plan_quench, quench_steps

    config_file = Path(request['config_file'])
    _emit('progress', message='building the crystal slab')
    settings = load_settings(settings_file=request.get('settings_file'))
    section = parse_config(config_file).get('quench')
    if section is None:
        return {'ok': False, 'error': 'the config has no [quench] section'}
    plan = plan_quench(section, settings, config_file.parent)
    chosen = (plan.members or [member for member, _block in plan.present])
    if not chosen:
        return {'ok': False, 'error': 'the config describes no block'}
    member = chosen[0]
    root = write_quench_root([member], Path(request['output_root']), settings,
                             root_name='preview')
    block = root / member_rel_dir(member) / 'build' / f'{member.material}_crystal_block.lmp'
    _box, natoms = read_data_header(block)
    steps = quench_steps(member)
    total = len(plan.members) + len(plan.present)
    label = (f"Crystal slab of {member.material} [{member.key.key8}]"
             f"{f' (1 of {total})' if total > 1 else ''}: about {natoms:,} atoms, "
             f"{steps:,} MD steps. Nothing is melted.")
    return {
        'ok': True, 'crystal_block': str(block), 'material': member.material,
        'elements': list(cifread(member.cif)['elements']), 'natoms': natoms,
        'steps': steps, 'label': label, 'warnings': [],
    }


def run(request: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a single-member package for preview."""
    if request['model'] == 'quench':
        return run_quench(request)
    from ..run.artifacts import discover_preview_target, generate_package

    config_file = request['config_file']
    model = request['model']
    output_root = Path(request['output_root'])
    layer = request.get('layer')

    _emit('progress', message=f"building {model}")

    from ..builders.construction.amorphous import AmorphousMissing

    try:
        # No build path melt-quenches: a missing amorphous block raises.
        result = generate_package(
            config_file, model,
            output_root=output_root,
            settings_file=request.get('settings_file'),
            generate_hpc=False,
            simulation_root_name='preview',
        )
    except AmorphousMissing as exc:
        return {'ok': False, 'error': str(exc), **exc.as_payload()}

    if not result.created:
        return {
            'ok': False,
            'error': (result.failures[0].render() if result.failures
                      else 'nothing was generated'),
        }

    target = discover_preview_target(result.simulation_root, result.created[0], layer)
    if target is None:
        return {'ok': False, 'error': 'no assembly script was produced'}

    _emit('progress', message='parsing generated system')

    return {
        'ok': True,
        'simulation_root': str(target.simulation_root),
        'run_dir': str(target.run_dir),
        'init_script': str(target.init_script),
        'settings_file': str(target.settings_file) if target.settings_file else None,
        'layer': target.layer,
        'warnings': [f.render() for f in result.failures],
    }


def main(argv=None) -> int:
    """`python -m frictionsim2d.gui.preview_worker <request.json>`."""
    return run_worker(run, argv)


if __name__ == '__main__':
    sys.exit(main())
