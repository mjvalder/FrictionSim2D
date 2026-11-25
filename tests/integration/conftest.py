"""Shared build cache for the integration tier.

``generate_package`` is the slow part of every test here, so each example config
is generated at most once per session and every module reads that one tree.
"""
from typing import Dict

import pytest

from tests.helpers.examples import FROZEN_EXAMPLES, Example, build, build_example_file


@pytest.fixture(scope='session', name='build_example')
def _build_example(tmp_path_factory):
    """Return ``build_example(example) -> GenerationResult``, memoised on ``rel``."""
    cache: Dict[str, object] = {}

    def build_example(example: Example):
        if example.rel not in cache:
            cache[example.rel] = build_example_file(
                example, tmp_path_factory.mktemp(example.model))
        return cache[example.rel]

    return build_example


@pytest.fixture(scope='session', name='build_frozen')
def _build_frozen(tmp_path_factory):
    """Return ``build_frozen(name, model) -> GenerationResult`` for a
    ``tests/fixtures/examples/`` config, memoised: tests asserting exact values."""
    cache: Dict[str, object] = {}

    def build_frozen(name: str, model: str):
        if name not in cache:
            cache[name] = build(name, model, tmp_path_factory.mktemp(model),
                                source=FROZEN_EXAMPLES)
        return cache[name]

    return build_frozen
