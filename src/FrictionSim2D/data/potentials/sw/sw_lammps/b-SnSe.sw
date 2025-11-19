# by Jin-Wu Jiang, jwjiang5918@hotmail.com; jiangjinwu@shu.edu.cn

# SW parameters for b-SnSe    , used by LAMMPS
# these entries are in LAMMPS "metal" units:
#   epsilon = eV; sigma = Angstroms
#   other quantities are unitless

# format of a single entry (one or more lines):
#   element 1, element 2, element 3,
#   epsilon, sigma, a, lambda, gamma, costheta0, A, B, p, q,tol

# sw2 and sw3
Sn  Se  Se     1.000   1.510   2.494  26.294   1.000  -0.016   7.976   5.482  4  0   0.000
Se  Sn  Sn     1.000   1.510   2.494  26.294   1.000  -0.016   7.976   5.482  4  0   0.000

# zero terms
Sn  Sn  Sn     0.000   1.000   1.000   1.000   1.000   1.000   1.000   1.000  4  0   0.000
Sn  Sn  Se     0.000   1.000   1.000   1.000   1.000   1.000   1.000   1.000  4  0   0.000
Sn  Se  Sn     0.000   1.000   1.000   1.000   1.000   1.000   1.000   1.000  4  0   0.000
Se  Sn  Se     0.000   1.000   1.000   1.000   1.000   1.000   1.000   1.000  4  0   0.000
Se  Se  Sn     0.000   1.000   1.000   1.000   1.000   1.000   1.000   1.000  4  0   0.000
Se  Se  Se     0.000   1.000   1.000   1.000   1.000   1.000   1.000   1.000  4  0   0.000
