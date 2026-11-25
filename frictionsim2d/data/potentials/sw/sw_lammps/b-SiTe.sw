# by Jin-Wu Jiang, jwjiang5918@hotmail.com; jiangjinwu@shu.edu.cn

# SW parameters for b-SiTe    , used by LAMMPS
# these entries are in LAMMPS "metal" units:
#   epsilon = eV; sigma = Angstroms
#   other quantities are unitless

# format of a single entry (one or more lines):
#   element 1, element 2, element 3,
#   epsilon, sigma, a, lambda, gamma, costheta0, A, B, p, q,tol

# sw2 and sw3
Si  Te  Te     1.000   1.473   2.502  41.952   1.000  -0.014   9.285   5.567  4  0   0.000
Te  Si  Si     1.000   1.473   2.502  41.952   1.000  -0.014   9.285   5.567  4  0   0.000

# zero terms
Si  Si  Si     0.000   1.000   1.000   1.000   1.000   1.000   1.000   1.000  4  0   0.000
Si  Si  Te     0.000   1.000   1.000   1.000   1.000   1.000   1.000   1.000  4  0   0.000
Si  Te  Si     0.000   1.000   1.000   1.000   1.000   1.000   1.000   1.000  4  0   0.000
Te  Si  Te     0.000   1.000   1.000   1.000   1.000   1.000   1.000   1.000  4  0   0.000
Te  Te  Si     0.000   1.000   1.000   1.000   1.000   1.000   1.000   1.000  4  0   0.000
Te  Te  Te     0.000   1.000   1.000   1.000   1.000   1.000   1.000   1.000  4  0   0.000
