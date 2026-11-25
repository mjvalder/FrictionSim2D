# by Jin-Wu Jiang, jwjiang5918@hotmail.com; jiangjinwu@shu.edu.cn

# SW parameters for b-GeTe    , used by LAMMPS
# these entries are in LAMMPS "metal" units:
#   epsilon = eV; sigma = Angstroms
#   other quantities are unitless

# format of a single entry (one or more lines):
#   element 1, element 2, element 3,
#   epsilon, sigma, a, lambda, gamma, costheta0, A, B, p, q,tol

# sw2 and sw3
Ge  Te  Te     1.000   1.513   2.506  33.832   1.000  -0.013   9.704   5.605  4  0   0.000
Te  Ge  Ge     1.000   1.513   2.506  33.832   1.000  -0.013   9.704   5.605  4  0   0.000

# zero terms
Ge  Ge  Ge     0.000   1.000   1.000   1.000   1.000   1.000   1.000   1.000  4  0   0.000
Ge  Ge  Te     0.000   1.000   1.000   1.000   1.000   1.000   1.000   1.000  4  0   0.000
Ge  Te  Ge     0.000   1.000   1.000   1.000   1.000   1.000   1.000   1.000  4  0   0.000
Te  Ge  Te     0.000   1.000   1.000   1.000   1.000   1.000   1.000   1.000  4  0   0.000
Te  Te  Ge     0.000   1.000   1.000   1.000   1.000   1.000   1.000   1.000  4  0   0.000
Te  Te  Te     0.000   1.000   1.000   1.000   1.000   1.000   1.000   1.000  4  0   0.000
