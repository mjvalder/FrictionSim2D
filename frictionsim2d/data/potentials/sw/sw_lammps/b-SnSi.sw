# by Jin-Wu Jiang, jwjiang5918@hotmail.com; jiangjinwu@shu.edu.cn

# SW parameters for b-SnSi    , used by LAMMPS
# these entries are in LAMMPS "metal" units:
#   epsilon = eV; sigma = Angstroms
#   other quantities are unitless

# format of a single entry (one or more lines):
#   element 1, element 2, element 3,
#   epsilon, sigma, a, lambda, gamma, costheta0, A, B, p, q,tol

# sw2 and sw3
Sn  Si  Si     1.000   2.260   1.643  75.415   1.000  -0.396  16.463   0.773  4  0   0.000
Si  Sn  Sn     1.000   2.260   1.643  75.415   1.000  -0.396  16.463   0.773  4  0   0.000

# zero terms
Sn  Sn  Sn     0.000   1.000   1.000   1.000   1.000   1.000   1.000   1.000  4  0   0.000
Sn  Sn  Si     0.000   1.000   1.000   1.000   1.000   1.000   1.000   1.000  4  0   0.000
Sn  Si  Sn     0.000   1.000   1.000   1.000   1.000   1.000   1.000   1.000  4  0   0.000
Si  Sn  Si     0.000   1.000   1.000   1.000   1.000   1.000   1.000   1.000  4  0   0.000
Si  Si  Sn     0.000   1.000   1.000   1.000   1.000   1.000   1.000   1.000  4  0   0.000
Si  Si  Si     0.000   1.000   1.000   1.000   1.000   1.000   1.000   1.000  4  0   0.000
