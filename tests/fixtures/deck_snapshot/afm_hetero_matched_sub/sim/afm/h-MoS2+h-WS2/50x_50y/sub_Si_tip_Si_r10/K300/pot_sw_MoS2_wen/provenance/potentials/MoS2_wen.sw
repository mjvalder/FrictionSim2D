# Stillinger-Weber potential for MoS2 for LAMMPS
#
# Original parameters from:
# M. Wen, et al., J. Appl. Phys., 122, 244301 (2017).
#
# Converted from KIM format to LAMMPS sw format.
# This version includes all 8 triplet permutations for maximum compatibility.
#
# ASSUMPTIONS AND LIMITATIONS:
# 1. The original potential's three-body cutoff term (cutoff_jk) is ignored
#    as it is not supported by the standard LAMMPS 'pair_style sw'.
# 2. 'epsilon' is 1.0 as KIM energy parameters are already in eV.
# 3. 'sigma', 'a', and 'gamma' are derived from the central atom and pair data.
#
# Format: i j k epsilon sigma a lambda gamma cos(theta0) A B p q tol
#
# Mo-centered triplets
Mo Mo Mo  1.0  2.85295  1.94416  0.00000  0.47552  0.142857  3.97818  0.44460  5  0  0.0
Mo Mo S   1.0  2.85295  1.94416  0.00000  0.47552  0.142857  3.97818  0.44460  5  0  0.0
Mo S Mo   1.0  2.85295  1.94416  0.00000  0.47552  0.142857  3.97818  0.44460  5  0  0.0
Mo S S   1.0  2.85295  1.41150  7.47675  0.47552  0.142857  11.37974 0.52667  5  0  0.0
# S-centered triplets
S S S    1.0  2.84133  1.59065  0.00000  0.47746  0.142857  1.19074  0.90152  5  0  0.0
S S Mo   1.0  2.84133  1.59065  0.00000  0.47746  0.142857  1.19074  0.90152  5  0  0.0
S Mo S   1.0  2.84133  1.59065  0.00000  0.47746  0.142857  1.19074  0.90152  5  0  0.0
S Mo Mo   1.0  2.84133  1.41726  8.15952  0.47746  0.142857  11.37974 0.52667  5  0  0.0