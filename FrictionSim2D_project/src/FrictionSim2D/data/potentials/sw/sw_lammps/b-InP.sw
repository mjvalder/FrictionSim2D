# by Jin-Wu Jiang, jwjiang5918@hotmail.com; jiangjinwu@shu.edu.cn

# SW parameters for b-InP     , used by LAMMPS
# these entries are in LAMMPS "metal" units:
#   epsilon = eV; sigma = Angstroms
#   other quantities are unitless

# format of a single entry (one or more lines):
#   element 1, element 2, element 3,
#   epsilon, sigma, a, lambda, gamma, costheta0, A, B, p, q,tol

# sw2 and sw3
In  P   P      1.000   2.306   1.583  70.782   1.000  -0.437  20.610   0.648  4  0   0.000
P   In  In     1.000   2.306   1.583  70.782   1.000  -0.437  20.610   0.648  4  0   0.000

# zero terms
In  In  In     0.000   1.000   1.000   1.000   1.000   1.000   1.000   1.000  4  0   0.000
In  In  P      0.000   1.000   1.000   1.000   1.000   1.000   1.000   1.000  4  0   0.000
In  P   In     0.000   1.000   1.000   1.000   1.000   1.000   1.000   1.000  4  0   0.000
P   In  P      0.000   1.000   1.000   1.000   1.000   1.000   1.000   1.000  4  0   0.000
P   P   In     0.000   1.000   1.000   1.000   1.000   1.000   1.000   1.000  4  0   0.000
P   P   P      0.000   1.000   1.000   1.000   1.000   1.000   1.000   1.000  4  0   0.000
