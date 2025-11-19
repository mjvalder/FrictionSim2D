# by Jin-Wu Jiang, jwjiang5918@hotmail.com; jiangjinwu@shu.edu.cn

# SW parameters for b-GeSe    , used by LAMMPS
# these entries are in LAMMPS "metal" units:
#   epsilon = eV; sigma = Angstroms
#   other quantities are unitless

# format of a single entry (one or more lines):
#   element 1, element 2, element 3,
#   epsilon, sigma, a, lambda, gamma, costheta0, A, B, p, q,tol

# sw2 and sw3
Ge  Se  Se     1.000   1.430   2.466  34.791   1.000  -0.025   8.498   5.205  4  0   0.000
Se  Ge  Ge     1.000   1.430   2.466  34.791   1.000  -0.025   8.498   5.205  4  0   0.000

# zero terms
Ge  Ge  Ge     0.000   1.000   1.000   1.000   1.000   1.000   1.000   1.000  4  0   0.000
Ge  Ge  Se     0.000   1.000   1.000   1.000   1.000   1.000   1.000   1.000  4  0   0.000
Ge  Se  Ge     0.000   1.000   1.000   1.000   1.000   1.000   1.000   1.000  4  0   0.000
Se  Ge  Se     0.000   1.000   1.000   1.000   1.000   1.000   1.000   1.000  4  0   0.000
Se  Se  Ge     0.000   1.000   1.000   1.000   1.000   1.000   1.000   1.000  4  0   0.000
Se  Se  Se     0.000   1.000   1.000   1.000   1.000   1.000   1.000   1.000  4  0   0.000
