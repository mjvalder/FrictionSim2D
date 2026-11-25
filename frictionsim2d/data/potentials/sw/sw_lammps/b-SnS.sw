# by Jin-Wu Jiang, jwjiang5918@hotmail.com; jiangjinwu@shu.edu.cn

# SW parameters for b-SnS     , used by LAMMPS
# these entries are in LAMMPS "metal" units:
#   epsilon = eV; sigma = Angstroms
#   other quantities are unitless

# format of a single entry (one or more lines):
#   element 1, element 2, element 3,
#   epsilon, sigma, a, lambda, gamma, costheta0, A, B, p, q,tol

# sw2 and sw3
Sn  S   S      1.000   1.472   2.444  27.243   1.000  -0.031   7.392   4.994  4  0   0.000
S   Sn  Sn     1.000   1.472   2.444  27.243   1.000  -0.031   7.392   4.994  4  0   0.000

# zero terms
Sn  Sn  Sn     0.000   1.000   1.000   1.000   1.000   1.000   1.000   1.000  4  0   0.000
Sn  Sn  S      0.000   1.000   1.000   1.000   1.000   1.000   1.000   1.000  4  0   0.000
Sn  S   Sn     0.000   1.000   1.000   1.000   1.000   1.000   1.000   1.000  4  0   0.000
S   Sn  S      0.000   1.000   1.000   1.000   1.000   1.000   1.000   1.000  4  0   0.000
S   S   Sn     0.000   1.000   1.000   1.000   1.000   1.000   1.000   1.000  4  0   0.000
S   S   S      0.000   1.000   1.000   1.000   1.000   1.000   1.000   1.000  4  0   0.000
