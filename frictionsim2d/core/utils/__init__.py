"""Shared helpers, split by job so callers pay only for what they use.

Intentionally empty, with no re-exports: ``structures`` reaches ASE (~270 ms)
while ``paths`` and ``tokens`` are nearly free, and an eager re-export here
would hand every importer the ASE cost. Import the submodule you need, e.g.
``from ..core.utils.paths import get_material_path``;
``tests/repo/test_import_weight.py`` pins this.
"""
