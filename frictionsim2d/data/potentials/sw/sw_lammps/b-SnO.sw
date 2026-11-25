# by Jin-Wu Jiang, jwjiang5918@hotmail.com; jiangjinwu@shu.edu.cn

# SW parameters for b-SnO     , used by LAMMPS
# these entries are in LAMMPS "metal" units:
#   epsilon = eV; sigma = Angstroms
#   other quantities are unitless

# format of a single entry (one or more lines):
#   element 1, element 2, element 3,
#   epsilon, sigma, a, lambda, gamma, costheta0, A, B, p, q,tol

# sw2 and sw3
Sn  O   O      1.000   1.609   1.955  52.875   1.000  -0.219   9.133   1.760  4  0   0.000
O   Sn  Sn     1.000   1.609   1.955  52.875   1.000  -0.219   9.133   1.760  4  0   0.000

# zero terms
Sn  Sn  Sn     0.000   1.000   1.000   1.000   1.000   1.000   1.000   1.000  4  0   0.000
Sn  Sn  O      0.000   1.000   1.000   1.000   1.000   1.000   1.000   1.000  4  0   0.000
Sn  O   Sn     0.000   1.000   1.000   1.000   1.000   1.000   1.000   1.000  4  0   0.000
O   Sn  O      0.000   1.000   1.000   1.000   1.000   1.000   1.000   1.000  4  0   0.000
O   O   Sn     0.000   1.000   1.000   1.000   1.000   1.000   1.000   1.000  4  0   0.000
O   O   O      0.000   1.000   1.000   1.000   1.000   1.000   1.000   1.000  4  0   0.000
