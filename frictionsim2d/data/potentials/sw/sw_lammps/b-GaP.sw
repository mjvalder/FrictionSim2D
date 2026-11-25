# by Jin-Wu Jiang, jwjiang5918@hotmail.com; jiangjinwu@shu.edu.cn

# SW parameters for b-GaP     , used by LAMMPS
# these entries are in LAMMPS "metal" units:
#   epsilon = eV; sigma = Angstroms
#   other quantities are unitless

# format of a single entry (one or more lines):
#   element 1, element 2, element 3,
#   epsilon, sigma, a, lambda, gamma, costheta0, A, B, p, q,tol

# sw2 and sw3
Ga  P   P      1.000   2.152   1.557  95.438   1.000  -0.456  21.948   0.597  4  0   0.000
P   Ga  Ga     1.000   2.152   1.557  95.438   1.000  -0.456  21.948   0.597  4  0   0.000

# zero terms
Ga  Ga  Ga     0.000   1.000   1.000   1.000   1.000   1.000   1.000   1.000  4  0   0.000
Ga  Ga  P      0.000   1.000   1.000   1.000   1.000   1.000   1.000   1.000  4  0   0.000
Ga  P   Ga     0.000   1.000   1.000   1.000   1.000   1.000   1.000   1.000  4  0   0.000
P   Ga  P      0.000   1.000   1.000   1.000   1.000   1.000   1.000   1.000  4  0   0.000
P   P   Ga     0.000   1.000   1.000   1.000   1.000   1.000   1.000   1.000  4  0   0.000
P   P   P      0.000   1.000   1.000   1.000   1.000   1.000   1.000   1.000  4  0   0.000
