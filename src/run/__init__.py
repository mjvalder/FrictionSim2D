"""Running simulations: expansion, naming, HPC dispatch, the runner itself.

Intentionally free of eager re-exports. simulate.py imports the builders, and
builders import core.config and core.utils -- an eager __init__ here would resolve
that chain against a half-initialised package. Import the submodules directly.
"""
