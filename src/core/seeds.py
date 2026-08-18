"""Deterministic per-run seed derivation (spec §5 F5).

seed = H(base, material, layer, load, angle, seed_index, channel).

Including load and angle is load-bearing: sharing a seed across loads or
angles within one material shares the literal initial realization (identical
system, identical RNG stream). Sharing integers across *materials* is harmless
— different atom counts and type maps give uncorrelated streams — but the hash
includes material anyway for auditability.
"""
import hashlib

_LAMMPS_SEED_MAX = 900_000_000  # LAMMPS requires 0 < seed <= ~2^31


def derive_seed(base: int, *, material: str, layer: int,
                load: float | None, angle: float | None,
                seed_index: int, channel: str) -> int:
    """Pure function of (base, run identity, channel) -> LAMMPS-valid seed."""
    key = f"{base}|{material}|{layer}|{load!r}|{angle!r}|{seed_index}|{channel}"
    digest = hashlib.sha256(key.encode()).digest()
    return int.from_bytes(digest[:8], 'big') % _LAMMPS_SEED_MAX + 1
