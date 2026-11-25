# by Jin-Wu Jiang, jwjiang5918@hotmail.com; jiangjinwu@shu.edu.cn

# SW parameters for b-GeS     , used by LAMMPS
# these entries are in LAMMPS "metal" units:
#   epsilon = eV; sigma = Angstroms
#   other quantities are unitless

# format of a single entry (one or more lines):
#   element 1, element 2, element 3,
#   epsilon, sigma, a, lambda, gamma, costheta0, A, B, p, q,tol

# sw2 and sw3
Ge  S   S      1.000   1.363   2.448  35.249   1.000  -0.030   7.657   5.030  4  0   0.000
S   Ge  Ge     1.000   1.363   2.448  35.249   1.000  -0.030   7.657   5.030  4  0   0.000

# zero terms
Ge  Ge  Ge     0.000   1.000   1.000   1.000   1.000   1.000   1.000   1.000  4  0   0.000
Ge  Ge  S      0.000   1.000   1.000   1.000   1.000   1.000   1.000   1.000  4  0   0.000
Ge  S   Ge     0.000   1.000   1.000   1.000   1.000   1.000   1.000   1.000  4  0   0.000
S   Ge  S      0.000   1.000   1.000   1.000   1.000   1.000   1.000   1.000  4  0   0.000
S   S   Ge     0.000   1.000   1.000   1.000   1.000   1.000   1.000   1.000  4  0   0.000
S   S   S      0.000   1.000   1.000   1.000   1.000   1.000   1.000   1.000  4  0   0.000
