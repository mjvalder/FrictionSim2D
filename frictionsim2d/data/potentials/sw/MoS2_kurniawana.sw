# format of a single entry (one or more lines):
#   element 1, element 2, element 3, 
#   epsilon, sigma, a, lambda, gamma, costheta0, A, B, p, q, tol

Mo  Mo   18.431006000000014  0.006417863210000008   4.737178129999999 0   6.1694045400000075  1.538004999999998  5.54660
Mo  S     8.838613050000008    1.0479360300000011   8.266217440000004 0   1.9296799099999997  1.538004999999998  4.02692
S   S   0.37463396299999996     561.4292700000284  2.6619691299999673 0  0.41904814400000867  1.538004999999998  4.51956

Mo  S   S    4.287840759999989  0.1428569579923222  3.86095
S   Mo  Mo  14.428502599999979  0.1428569579923222  5.54660


#
# Format of parameter file:
#
# First line: number of species
#
# Lines 2~4: parameters for 2-body interaction
#   species1 species2 A B p q sigma gamma cutoff
#
# Lines 5~6: parameters for 3-body interaction
#   species1 species2 species3 lambda cos_beta0 cutoff_jk
#
# species is valid KIM API particle species string
# A and lambda in [eV]
# sigma, gamma, cutoff, and cutoff_jk in [Angstrom]
# others are unitless
#
# Kurniawan