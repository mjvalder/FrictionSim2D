"""Geometry measurements shared by the flake-path tests."""
import numpy as np

# --- body moved verbatim from tests/lattice/test_lattice_apply.py ---


def _flake_lattice_orientation(atoms):
    """Return the flake lattice's in-plane bond orientation, folded to [0, 60).

    Measures nearest-neighbour bond directions off a FINITE flake's atom xy
    positions and averages them on the circle after folding into the hexagonal
    60 deg wedge. Twisting the flake by theta shifts this value by theta (mod
    60), so comparing two builds' orientations recovers the applied twist.
    Factored module-level so the D3b flake-path tests (test_flake_slide.py,
    test_hetero_afm.py) can import the same measurement.
    """
    from scipy.spatial import cKDTree
    xy = atoms.get_positions()[:, :2]
    # Merge coincident xy columns before measuring: a TMD flake (MoS2/WS2) has
    # its two chalcogen atoms stacked in a z-column at the SAME xy, so the raw
    # xy projection has zero-distance pairs that would collapse the NN metric.
    xy = np.unique(np.round(xy, 1), axis=0)
    t = cKDTree(xy)
    d, _ = t.query(xy, k=2)
    nn = np.median(d[:, 1])
    angs = []
    for i, nb in enumerate(t.query_ball_point(xy, nn * 1.15)):
        for j in nb:
            if j == i:
                continue
            v = xy[j] - xy[i]
            if np.linalg.norm(v) < nn * 0.5:
                continue
            angs.append(np.degrees(np.arctan2(v[1], v[0])) % 60.0)
    r = np.array(angs) / 60.0 * 2 * np.pi
    return (np.degrees(np.arctan2(np.sin(r).mean(), np.cos(r).mean())) / 360 * 60.0) % 60.0
