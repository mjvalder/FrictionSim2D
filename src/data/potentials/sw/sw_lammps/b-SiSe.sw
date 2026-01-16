# by Jin-Wu Jiang, jwjiang5918@hotmail.com; jiangjinwu@shu.edu.cn

# SW parameters for b-SiSe    , used by LAMMPS
# these entries are in LAMMPS "metal" units:
#   epsilon = eV; sigma = Angstroms
#   other quantities are unitless

# format of a single entry (one or more lines):
#   element 1, element 2, element 3,
#   epsilon, sigma, a, lambda, gamma, costheta0, A, B, p, q,tol

# sw2 and sw3
Si  Se  Se     1.000   1.349   2.514  45.968   1.000  -0.010   7.857   5.683  4  0   0.000
Se  Si  Si     1.000   1.349   2.514  45.968   1.000  -0.010   7.857   5.683  4  0   0.000

# zero terms
Si  Si  Si     0.000   1.000   1.000   1.000   1.000   1.000   1.000   1.000  4  0   0.000
Si  Si  Se     0.000   1.000   1.000   1.000   1.000   1.000   1.000   1.000  4  0   0.000
Si  Se  Si     0.000   1.000   1.000   1.000   1.000   1.000   1.000   1.000  4  0   0.000
Se  Si  Se     0.000   1.000   1.000   1.000   1.000   1.000   1.000   1.000  4  0   0.000
Se  Se  Si     0.000   1.000   1.000   1.000   1.000   1.000   1.000   1.000  4  0   0.000
Se  Se  Se     0.000   1.000   1.000   1.000   1.000   1.000   1.000   1.000  4  0   0.000
