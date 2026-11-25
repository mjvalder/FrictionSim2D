# by Jin-Wu Jiang, jwjiang5918@hotmail.com; jiangjinwu@shu.edu.cn

# SW parameters for b-CSe     , used by LAMMPS
# these entries are in LAMMPS "metal" units:
#   epsilon = eV; sigma = Angstroms
#   other quantities are unitless

# format of a single entry (one or more lines):
#   element 1, element 2, element 3,
#   epsilon, sigma, a, lambda, gamma, costheta0, A, B, p, q,tol

# sw2 and sw3
C   Se  Se     1.000   1.298   2.212  61.215   1.000  -0.111   7.691   3.137  4  0   0.000
Se  C   C      1.000   1.298   2.212  61.215   1.000  -0.111   7.691   3.137  4  0   0.000

# zero terms
C   C   C      0.000   1.000   1.000   1.000   1.000   1.000   1.000   1.000  4  0   0.000
C   C   Se     0.000   1.000   1.000   1.000   1.000   1.000   1.000   1.000  4  0   0.000
C   Se  C      0.000   1.000   1.000   1.000   1.000   1.000   1.000   1.000  4  0   0.000
Se  C   Se     0.000   1.000   1.000   1.000   1.000   1.000   1.000   1.000  4  0   0.000
Se  Se  C      0.000   1.000   1.000   1.000   1.000   1.000   1.000   1.000  4  0   0.000
Se  Se  Se     0.000   1.000   1.000   1.000   1.000   1.000   1.000   1.000  4  0   0.000
