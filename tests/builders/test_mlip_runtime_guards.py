"""Guards for the two production-scale failures found by generate->run (F2, F3).

Both were invisible to component-level tests: the emitted deck was individually
valid, and only failed when a real builder output met a real solver.
"""

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
TEMPLATES = REPO / 'src' / 'templates'

#: Every template that emits a tiled decomposition / RCB balance.
TILED_TEMPLATES = [
    'sheetonsheet/slide.lmp',
    'sheetonsheet/pes_scan.lmp',
    'hetero/slide.lmp',
    'flake/slide.lmp',
    'afm/pes_scan.lmp',
]


@pytest.mark.parametrize('rel', TILED_TEMPLATES)
def test_tiled_comm_is_gated_for_mlip_sheets(rel):
    """F2: `comm_style tiled` breaks pair_deepmd, so it must be conditional.

    Measured 2026-08-11: with tiled, pair_deepmd tries to allocate ~558 GB on a
    270-atom cell (`DefaultCPUAllocator`, host side -- not a GPU limit, and the
    size varies run to run, the signature of a bogus length). Removing
    `balance` alone does NOT help; removing `comm_style tiled` alone does, and
    the identical deck then runs. tiled/RCB buy nothing at our design point
    anyway (mpi_ranks == 1 per GPU).
    """
    text = (TEMPLATES / rel).read_text(encoding='utf-8')
    for keyword in ('comm_style', 'balance'):
        for i, line in enumerate(text.splitlines()):
            stripped = line.strip()
            if not stripped.startswith(keyword):
                continue
            window = '\n'.join(text.splitlines()[max(0, i - 10):i])
            assert 'mlip_sheet' in window, (
                f"{rel}:{i+1} emits {stripped!r} unguarded; it must sit inside "
                f"a Jinja block keyed on `mlip_sheet` so MLIP runs get the "
                f"default brick decomposition"
            )


def test_d3_cutoff_vs_box_is_validated():
    """F3: d3's 16 A cutoff needs a box bigger than cutoff + neighbour skin.

    Measured: an x=y=15 sheet builds a 15.9 x 16.5 A box and dies with
    "Communication cutoff ... cannot exceed periodic box length"; x=y=20
    (19.1 x 22.1 A) clears it. Small screening cells therefore need a smaller
    d3_cutoff, and the failure must be diagnosed up front rather than deep in a
    LAMMPS run.
    """
    from src.core.mlip import check_d3_fits_box

    # 16.0 cutoff + 0.3 skin = 16.3 required
    with pytest.raises(ValueError, match='(?i)d3_cutoff'):
        check_d3_fits_box(box_x=15.9, box_y=16.5, d3_cutoff=16.0, skin=0.3)

    # comfortably larger box is fine
    check_d3_fits_box(box_x=19.1, box_y=22.1, d3_cutoff=16.0, skin=0.3)

    # a reduced cutoff rescues the small cell
    check_d3_fits_box(box_x=15.9, box_y=16.5, d3_cutoff=10.0, skin=0.3)

    # the message must say what to do about it
    with pytest.raises(ValueError, match='(?i)reduce|smaller|increase'):
        check_d3_fits_box(box_x=12.0, box_y=12.0, d3_cutoff=16.0, skin=0.3)
