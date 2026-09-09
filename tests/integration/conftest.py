"""Shared build cache for the integration tier.

``generate_package`` is the slow part of every test here, so each example config
is generated at most once per session and every module reads that one tree.
"""
from typing import Dict

import pytest

from tests.helpers.examples import build


@pytest.fixture(scope='session', name='build_example')
def _build_example(tmp_path_factory):
    """Return ``build_example(name, model) -> GenerationResult``, memoised."""
    cache: Dict[str, object] = {}

    def build_example(name: str, model: str):
        if name not in cache:
            cache[name] = build(name, model, tmp_path_factory.mktemp(model))
        return cache[name]

    return build_example
