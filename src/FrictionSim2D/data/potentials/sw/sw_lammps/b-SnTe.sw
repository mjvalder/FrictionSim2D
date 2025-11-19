# by Jin-Wu Jiang, jwjiang5918@hotmail.com; jiangjinwu@shu.edu.cn

# SW parameters for b-SnTe    , used by LAMMPS
# these entries are in LAMMPS "metal" units:
#   epsilon = eV; sigma = Angstroms
#   other quantities are unitless

# format of a single entry (one or more lines):
#   element 1, element 2, element 3,
#   epsilon, sigma, a, lambda, gamma, costheta0, A, B, p, q,tol

# sw2 and sw3
Sn  Te  Te     1.000   1.559   2.577  24.867   1.000   0.008   8.864   6.378  4  0   0.000
Te  Sn  Sn     1.000   1.559   2.577  24.867   1.000   0.008   8.864   6.378  4  0   0.000

# zero terms
Sn  Sn  Sn     0.000   1.000   1.000   1.000   1.000   1.000   1.000   1.000  4  0   0.000
Sn  Sn  Te     0.000   1.000   1.000   1.000   1.000   1.000   1.000   1.000  4  0   0.000
Sn  Te  Sn     0.000   1.000   1.000   1.000   1.000   1.000   1.000   1.000  4  0   0.000
Te  Sn  Te     0.000   1.000   1.000   1.000   1.000   1.000   1.000   1.000  4  0   0.000
Te  Te  Sn     0.000   1.000   1.000   1.000   1.000   1.000   1.000   1.000  4  0   0.000
Te  Te  Te     0.000   1.000   1.000   1.000   1.000   1.000   1.000   1.000  4  0   0.000
