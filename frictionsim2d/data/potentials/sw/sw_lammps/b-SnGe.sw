# by Jin-Wu Jiang, jwjiang5918@hotmail.com; jiangjinwu@shu.edu.cn

# SW parameters for b-SnGe    , used by LAMMPS
# these entries are in LAMMPS "metal" units:
#   epsilon = eV; sigma = Angstroms
#   other quantities are unitless

# format of a single entry (one or more lines):
#   element 1, element 2, element 3,
#   epsilon, sigma, a, lambda, gamma, costheta0, A, B, p, q,tol

# sw2 and sw3
Sn  Ge  Ge     1.000   2.267   1.666  77.881   1.000  -0.380  13.674   0.826  4  0   0.000
Ge  Sn  Sn     1.000   2.267   1.666  77.881   1.000  -0.380  13.674   0.826  4  0   0.000

# zero terms
Sn  Sn  Sn     0.000   1.000   1.000   1.000   1.000   1.000   1.000   1.000  4  0   0.000
Sn  Sn  Ge     0.000   1.000   1.000   1.000   1.000   1.000   1.000   1.000  4  0   0.000
Sn  Ge  Sn     0.000   1.000   1.000   1.000   1.000   1.000   1.000   1.000  4  0   0.000
Ge  Sn  Ge     0.000   1.000   1.000   1.000   1.000   1.000   1.000   1.000  4  0   0.000
Ge  Ge  Sn     0.000   1.000   1.000   1.000   1.000   1.000   1.000   1.000  4  0   0.000
Ge  Ge  Ge     0.000   1.000   1.000   1.000   1.000   1.000   1.000   1.000  4  0   0.000
