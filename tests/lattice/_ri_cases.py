"""Deterministic inputs for the registry-index characterisation fixture."""
import numpy as np


def cases():
    """Return {name: (cart_a, cart_b, cell, n_grid)} for three reference cells."""
    out = {}
    a = 2.46
    cell = np.array([[a, 0.0], [-a / 2, a * np.sqrt(3) / 2]])
    cart = np.array([[0.0, 0.0], [1 / 3, 2 / 3]]) @ cell
    out["graphene_bilayer"] = (cart, cart, cell, 12)

    a = 3.16
    uc = np.array([[a, 0.0], [-a / 2, a * np.sqrt(3) / 2]])
    base = np.array([[0.0, 0.0], [1 / 3, 2 / 3], [1 / 3, 2 / 3]]) @ uc
    n = 5
    shifts = np.array([[i, j] for i in range(n) for j in range(n)]) @ uc
    cart_a = (base[:, None, :] + shifts[None]).reshape(-1, 2)
    big = n * uc
    cart_b = ((cart_a @ np.linalg.inv(big)) % 1.0 * 1.0) @ big
    cart_b = cart_b * 0.99 + 0.07 * np.sin(cart_b)  # strained, mildly distorted partner
    cart_b = (cart_b @ np.linalg.inv(big) % 1.0) @ big
    out["mos2_hetero"] = (cart_a, cart_b, big, 15)

    lx, ly = 5 * 2.46, 3 * 4.26
    uc_r = np.array([[0, 0], [1.23, 0.71], [1.23, 2.13], [0.0, 2.84]])
    cell_r = np.array([[lx, 0.0], [0.0, ly]])
    pos = np.array(
        [p + np.array([i * 2.46, j * 4.26]) for i in range(5) for j in range(3) for p in uc_r]
    )
    pos_b = pos + np.array([0.31, 0.17])
    out["rectangular"] = (pos, pos_b, cell_r, 10)
    return out
