# by Jin-Wu Jiang, jwjiang5918@hotmail.com; jiangjinwu@shu.edu.cn

# SW parameters for b-InAs    , used by LAMMPS
# these entries are in LAMMPS "metal" units:
#   epsilon = eV; sigma = Angstroms
#   other quantities are unitless

# format of a single entry (one or more lines):
#   element 1, element 2, element 3,
#   epsilon, sigma, a, lambda, gamma, costheta0, A, B, p, q,tol

# sw2 and sw3
In  As  As     1.000   2.320   1.624  64.931   1.000  -0.409  18.099   0.730  4  0   0.000
As  In  In     1.000   2.320   1.624  64.931   1.000  -0.409  18.099   0.730  4  0   0.000

# zero terms
In  In  In     0.000   1.000   1.000   1.000   1.000   1.000   1.000   1.000  4  0   0.000
In  In  As     0.000   1.000   1.000   1.000   1.000   1.000   1.000   1.000  4  0   0.000
In  As  In     0.000   1.000   1.000   1.000   1.000   1.000   1.000   1.000  4  0   0.000
As  In  As     0.000   1.000   1.000   1.000   1.000   1.000   1.000   1.000  4  0   0.000
As  As  In     0.000   1.000   1.000   1.000   1.000   1.000   1.000   1.000  4  0   0.000
As  As  As     0.000   1.000   1.000   1.000   1.000   1.000   1.000   1.000  4  0   0.000
