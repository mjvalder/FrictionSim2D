# by Jin-Wu Jiang, jwjiang5918@hotmail.com; jiangjinwu@shu.edu.cn

# SW parameters for b-SiS     , used by LAMMPS
# these entries are in LAMMPS "metal" units:
#   epsilon = eV; sigma = Angstroms
#   other quantities are unitless

# format of a single entry (one or more lines):
#   element 1, element 2, element 3,
#   epsilon, sigma, a, lambda, gamma, costheta0, A, B, p, q,tol

# sw2 and sw3
Si  S   S      1.000   1.264   2.514  45.954   1.000  -0.010   6.897   5.687  4  0   0.000
S   Si  Si     1.000   1.264   2.514  45.954   1.000  -0.010   6.897   5.687  4  0   0.000

# zero terms
Si  Si  Si     0.000   1.000   1.000   1.000   1.000   1.000   1.000   1.000  4  0   0.000
Si  Si  S      0.000   1.000   1.000   1.000   1.000   1.000   1.000   1.000  4  0   0.000
Si  S   Si     0.000   1.000   1.000   1.000   1.000   1.000   1.000   1.000  4  0   0.000
S   Si  S      0.000   1.000   1.000   1.000   1.000   1.000   1.000   1.000  4  0   0.000
S   S   Si     0.000   1.000   1.000   1.000   1.000   1.000   1.000   1.000  4  0   0.000
S   S   S      0.000   1.000   1.000   1.000   1.000   1.000   1.000   1.000  4  0   0.000
