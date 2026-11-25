# by Jin-Wu Jiang, jwjiang5918@hotmail.com; jiangjinwu@shu.edu.cn

# SW parameters for b-CS      , used by LAMMPS
# these entries are in LAMMPS "metal" units:
#   epsilon = eV; sigma = Angstroms
#   other quantities are unitless

# format of a single entry (one or more lines):
#   element 1, element 2, element 3,
#   epsilon, sigma, a, lambda, gamma, costheta0, A, B, p, q,tol

# sw2 and sw3
C   S   S      1.000   1.233   2.142  61.413   1.000  -0.138   6.014   2.703  4  0   0.000
S   C   C      1.000   1.233   2.142  61.413   1.000  -0.138   6.014   2.703  4  0   0.000

# zero terms
C   C   C      0.000   1.000   1.000   1.000   1.000   1.000   1.000   1.000  4  0   0.000
C   C   S      0.000   1.000   1.000   1.000   1.000   1.000   1.000   1.000  4  0   0.000
C   S   C      0.000   1.000   1.000   1.000   1.000   1.000   1.000   1.000  4  0   0.000
S   C   S      0.000   1.000   1.000   1.000   1.000   1.000   1.000   1.000  4  0   0.000
S   S   C      0.000   1.000   1.000   1.000   1.000   1.000   1.000   1.000  4  0   0.000
S   S   S      0.000   1.000   1.000   1.000   1.000   1.000   1.000   1.000  4  0   0.000
