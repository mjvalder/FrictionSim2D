# by Jin-Wu Jiang, jwjiang5918@hotmail.com; jiangjinwu@shu.edu.cn

# SW parameters for b-CO      , used by LAMMPS
# these entries are in LAMMPS "metal" units:
#   epsilon = eV; sigma = Angstroms
#   other quantities are unitless

# format of a single entry (one or more lines):
#   element 1, element 2, element 3,
#   epsilon, sigma, a, lambda, gamma, costheta0, A, B, p, q,tol

# sw2 and sw3
C   O   O      1.000   1.054   2.175  65.778   1.000  -0.125   7.656   2.900  4  0   0.000
O   C   C      1.000   1.054   2.175  65.778   1.000  -0.125   7.656   2.900  4  0   0.000

# zero terms
C   C   C      0.000   1.000   1.000   1.000   1.000   1.000   1.000   1.000  4  0   0.000
C   C   O      0.000   1.000   1.000   1.000   1.000   1.000   1.000   1.000  4  0   0.000
C   O   C      0.000   1.000   1.000   1.000   1.000   1.000   1.000   1.000  4  0   0.000
O   C   O      0.000   1.000   1.000   1.000   1.000   1.000   1.000   1.000  4  0   0.000
O   O   C      0.000   1.000   1.000   1.000   1.000   1.000   1.000   1.000  4  0   0.000
O   O   O      0.000   1.000   1.000   1.000   1.000   1.000   1.000   1.000  4  0   0.000
