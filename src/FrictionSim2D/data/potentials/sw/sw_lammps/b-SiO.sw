# by Jin-Wu Jiang, jwjiang5918@hotmail.com; jiangjinwu@shu.edu.cn

# SW parameters for b-SiO     , used by LAMMPS
# these entries are in LAMMPS "metal" units:
#   epsilon = eV; sigma = Angstroms
#   other quantities are unitless

# format of a single entry (one or more lines):
#   element 1, element 2, element 3,
#   epsilon, sigma, a, lambda, gamma, costheta0, A, B, p, q,tol

# sw2 and sw3
Si  O   O      1.000   1.200   2.197  40.695   1.000  -0.116   5.819   3.043  4  0   0.000
O   Si  Si     1.000   1.200   2.197  40.695   1.000  -0.116   5.819   3.043  4  0   0.000

# zero terms
Si  Si  Si     0.000   1.000   1.000   1.000   1.000   1.000   1.000   1.000  4  0   0.000
Si  Si  O      0.000   1.000   1.000   1.000   1.000   1.000   1.000   1.000  4  0   0.000
Si  O   Si     0.000   1.000   1.000   1.000   1.000   1.000   1.000   1.000  4  0   0.000
O   Si  O      0.000   1.000   1.000   1.000   1.000   1.000   1.000   1.000  4  0   0.000
O   O   Si     0.000   1.000   1.000   1.000   1.000   1.000   1.000   1.000  4  0   0.000
O   O   O      0.000   1.000   1.000   1.000   1.000   1.000   1.000   1.000  4  0   0.000
