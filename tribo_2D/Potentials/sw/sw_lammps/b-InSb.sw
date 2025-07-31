# by Jin-Wu Jiang, jwjiang5918@hotmail.com; jiangjinwu@shu.edu.cn

# SW parameters for b-InSb    , used by LAMMPS
# these entries are in LAMMPS "metal" units:
#   epsilon = eV; sigma = Angstroms
#   other quantities are unitless

# format of a single entry (one or more lines):
#   element 1, element 2, element 3,
#   epsilon, sigma, a, lambda, gamma, costheta0, A, B, p, q,tol

# sw2 and sw3
In  Sb  Sb     1.000   2.445   1.650  61.578   1.000  -0.391  16.706   0.788  4  0   0.000
Sb  In  In     1.000   2.445   1.650  61.578   1.000  -0.391  16.706   0.788  4  0   0.000

# zero terms
In  In  In     0.000   1.000   1.000   1.000   1.000   1.000   1.000   1.000  4  0   0.000
In  In  Sb     0.000   1.000   1.000   1.000   1.000   1.000   1.000   1.000  4  0   0.000
In  Sb  In     0.000   1.000   1.000   1.000   1.000   1.000   1.000   1.000  4  0   0.000
Sb  In  Sb     0.000   1.000   1.000   1.000   1.000   1.000   1.000   1.000  4  0   0.000
Sb  Sb  In     0.000   1.000   1.000   1.000   1.000   1.000   1.000   1.000  4  0   0.000
Sb  Sb  Sb     0.000   1.000   1.000   1.000   1.000   1.000   1.000   1.000  4  0   0.000
