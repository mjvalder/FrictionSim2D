"""Generation-level coverage for the one MLIP topology that actually runs.

Every other MLIP test stops at a builder method — usually
``_generate_potentials`` — and asserts the text it produces. That leaves the
path a *user* takes completely uncovered: ``run afm/sheetonsheet/pes-sheet`` →
``run/simulate.py::run_simulations`` → sweep expansion → builder → model staging →
on-disk tree. The pieces this file pins live only in that seam:

* the model file is staged **once per campaign** into the shared
  ``<root>/potentials/`` directory and the deck references it by the relative
  ``potentials/<name>`` prefix, not by the caller's absolute path (which would
  break the moment the tree is rsynced to a cluster);
* the MLIP flavour token (``pot_deepmd-d3_<stem>``) reaches the directory name,
  so two dispersion settings of one material cannot overwrite each other;
* the shipped MLIP examples in ``examples/<mode>/`` files still generate.

The model file is faked. Nothing here runs LAMMPS — the point is the generated
tree, and a real DPA/MACE model is multi-GB and lives outside the repo.
"""

import os
from pathlib import Path

import pytest

from frictionsim2d.core.config import parse_config, write_config
from frictionsim2d.potentials.manager import stage_model_file
from frictionsim2d.run.simulate import run_simulations
from tests.helpers.examples import discover

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = REPO_ROOT / 'examples'

#: (example config relative to examples/, `run` model name): the discovered MLIP
#: examples that are single-component sheet cells. The AFM one is validated by
#: test_examples_mlip.py, not generated here (it is a known-invalid cell there).
MLIP_EXAMPLES = [(e.rel, e.model) for e in discover()
                 if e.mlip and e.model in ('sheetonsheet', 'pes_sheet')]


def _fake_model(tmp_path: Path, name: str = 'dpa_fake.pth') -> Path:
    """A stand-in for a frozen DeePMD model: binary, deliberately not UTF-8."""
    path = tmp_path / name
    path.write_bytes(b"\x80\x02not-a-real-model\xff")
    return path


def _campaign_root(run_dir: Path) -> Path:
    """Walk up from a leaf run directory to the `simulation_<timestamp>/` root.

    ``run_simulations`` returns the innermost run directory, but the shared
    model store lives at the campaign root -- that is the whole point of it.
    """
    for parent in run_dir.parents:
        if parent.name.startswith('simulation_'):
            return parent
    raise AssertionError(f"no simulation_* ancestor of {run_dir}")


def _config_pointing_at(example: str, tmp_path: Path) -> Path:
    """Rewrite a shipped example so `pot_path` names a local fake model."""
    raw = parse_config(EXAMPLES / example)
    raw['2D']['pot_path'] = str(_fake_model(tmp_path))
    return write_config(raw, tmp_path / Path(example).name)


@pytest.mark.parametrize('example, model', MLIP_EXAMPLES)
def test_shipped_mlip_example_generates_end_to_end(example, model, tmp_path,
                                                   monkeypatch):
    """`run <model> examples/<mode>/<mlip example>` must produce a tree, not a stack trace."""
    monkeypatch.chdir(REPO_ROOT)
    config_path = _config_pointing_at(example, tmp_path)

    created, _, _, _ = run_simulations(
        config_file=str(config_path), model=model, settings_file=None,
        output_root=tmp_path / 'out', generate_hpc=False,
    )

    assert len(created) == 1
    settings = list(Path(created[0]).rglob('system.in.settings'))
    assert settings, f"{model} generated no system.in.settings"


@pytest.mark.parametrize('example, model', MLIP_EXAMPLES)
def test_model_is_staged_once_at_the_campaign_root(example, model, tmp_path,
                                                   monkeypatch):
    """The deck must reference the SHARED `potentials/` copy, by relative path.

    ``PotentialManager._get_potential_path`` routes MLIP model files (matched by
    extension) to ``shared_potentials_dir`` rather than giving each run its own
    copy, because a foundation model is multi-GB and a campaign is many runs.
    The emitted path must therefore be the ``potentials/<name>`` prefix — an
    absolute generation-machine path would not survive the rsync to a cluster.
    """
    monkeypatch.chdir(REPO_ROOT)
    config_path = _config_pointing_at(example, tmp_path)

    created, _, _, _ = run_simulations(
        config_file=str(config_path), model=model, settings_file=None,
        output_root=tmp_path / 'out', generate_hpc=False,
    )
    run_dir = Path(created[0])
    campaign_root = _campaign_root(run_dir)
    deck = next(iter(run_dir.rglob('system.in.settings'))).read_text(encoding='utf-8')

    assert 'pair_style' in deck
    # Referenced by the shared relative prefix...
    assert 'potentials/dpa_fake.pth' in deck
    # ...and never by the absolute path the config gave us.
    assert str(tmp_path / 'dpa_fake.pth') not in deck
    # ...with exactly one shared copy for the whole campaign, at its root.
    assert (campaign_root / 'potentials' / 'dpa_fake.pth').is_file()


@pytest.mark.parametrize('example, model', MLIP_EXAMPLES)
def test_flavour_token_reaches_the_directory_name(example, model, tmp_path,
                                                  monkeypatch):
    """The MLIP flavour (pot_ segment) must actually partition the output tree.

    Without it a `dispersion = d3` run and a `dispersion = none` run of the same
    material at the same temperature land in one directory and clobber each
    other. Both shipped examples set `dispersion = d3`.
    """
    monkeypatch.chdir(REPO_ROOT)
    config_path = _config_pointing_at(example, tmp_path)

    created, _, _, _ = run_simulations(
        config_file=str(config_path), model=model, settings_file=None,
        output_root=tmp_path / 'out', generate_hpc=False,
    )
    rels = {p.parent.parent.relative_to(tmp_path / 'out')
            for p in Path(created[0]).rglob('system.in.settings')}

    assert rels, f"{model} generated no system.in.settings"
    for rel in rels:
        # K<temp> stays unsuffixed; the flavour is the pot_ segment below it.
        assert 'pot_deepmd-d3_dpa_fake' in rel.parts, rel
        assert 'K0' in rel.parts, rel


def test_the_deck_carries_a_full_element_list(tmp_path, monkeypatch):
    """The silent-wrong-physics guard, asserted on a real generated deck.

    ``pair_coeff * * deepmd`` with no element list does NOT error: LAMMPS maps
    the types onto the model's first N elements (H, He, Li, ...) and quietly
    returns the wrong energy (-61.89 eV against -256.90 eV on the reference
    MoS2 trilayer). One entry per atom type is what makes the run meaningful,
    so it is worth pinning outside the unit tests too.
    """
    monkeypatch.chdir(REPO_ROOT)
    config_path = _config_pointing_at('sheetonsheet/mlip_full.ini', tmp_path)

    created, _, _, _ = run_simulations(
        config_file=str(config_path), model='sheetonsheet', settings_file=None,
        output_root=tmp_path / 'out', generate_hpc=False,
    )
    deck = next(iter(Path(created[0]).rglob('system.in.settings')))
    line = next(ln for ln in deck.read_text(encoding='utf-8').splitlines()
                if ln.startswith('pair_coeff * * deepmd'))

    # 3 layers x {Mo, S} = 6 sheet types, PLUS the virtual drive atom, which
    # must NOT be left as NULL: pair_deepmd errors on it and dispersion/d3
    # rejects it outright, so it is mapped to the framework's placeholder
    # element (`MLIPSpec.virtual_atom_element`, spec section 1.4). Seven
    # entries, one per LAMMPS atom type.
    elements = line.split('#')[0].split()[4:]
    assert elements == ['Mo', 'Mo', 'Mo', 'S', 'S', 'S', 'Mo'], line
    # D3 must span exactly the same types, one real element each.
    assert 'pair_coeff * * dispersion/d3 Mo Mo Mo S S S Mo' in \
        deck.read_text(encoding='utf-8')


# ---------------------------------------------------------------------------
# Provenance staging: a real file per run, but one inode per campaign
# ---------------------------------------------------------------------------

def test_mlip_model_is_hard_linked_into_provenance(tmp_path):
    """N runs of a campaign must cost ONE model's worth of disk, not N.

    `PotentialManager._get_potential_path` already dedupes the *deck-referenced*
    copy into the campaign-root `potentials/`, but `add_to_provenance` used to
    copy the model again into every run's `provenance/potentials/`. With a
    multi-GB foundation model over a real sweep that dominates everything else
    on disk.

    The fix must not weaken provenance: each run still gets a genuine, openable
    file at its own path (the manifest's sha256 is what pins identity, and it is
    computed from the source either way) -- they simply share an inode.
    """
    src = tmp_path / 'dpa_fake.pth'
    src.write_bytes(b'\x80\x02model-bytes\xff' * 64)

    dests = []
    for i in range(5):
        d = tmp_path / f'run{i}' / 'provenance' / 'potentials'
        d.mkdir(parents=True)
        dest = d / src.name
        stage_model_file(src, dest)
        dests.append(dest)

    # Every run has a real file with the real bytes...
    assert all(d.is_file() and d.read_bytes() == src.read_bytes() for d in dests)
    # ...and they are all one inode, so the campaign pays for the model once.
    assert len({d.stat().st_ino for d in dests}) == 1


def test_classical_potentials_are_still_copied(tmp_path):
    """Regression guard: only MLIP model files change staging behaviour.

    An SW/Tersoff file is KB-scale text a user might reasonably expect to be
    independent per run, so it keeps the plain `shutil.copy2` it always had.
    """
    src = tmp_path / 'MoS2.sw'
    src.write_text('# stillinger-weber\n', encoding='utf-8')

    dests = []
    for i in range(3):
        d = tmp_path / f'run{i}' / 'provenance' / 'potentials'
        d.mkdir(parents=True)
        dest = d / src.name
        stage_model_file(src, dest)
        dests.append(dest)

    assert all(d.read_text(encoding='utf-8') == src.read_text(encoding='utf-8')
               for d in dests)
    assert len({d.stat().st_ino for d in dests}) == 3


def test_staging_falls_back_to_a_copy_when_linking_fails(tmp_path, monkeypatch):
    """A cross-device dest, or a filesystem without hard links, must still work.

    The fallback is what makes this change strictly a space optimisation: if it
    cannot link, it copies, and the result is byte-identical to the old
    behaviour. It must never turn an unlinkable path into a failed build.
    """
    src = tmp_path / 'dpa_fake.pth'
    src.write_bytes(b'\x80\x02model\xff')
    dest = tmp_path / 'out' / 'dpa_fake.pth'
    dest.parent.mkdir(parents=True)

    def _refuse(*_a, **_k):
        raise OSError(18, 'Invalid cross-device link')
    monkeypatch.setattr(os, 'link', _refuse)

    stage_model_file(src, dest)

    assert dest.read_bytes() == src.read_bytes()
    assert dest.stat().st_ino != src.stat().st_ino     # a copy, not a link
