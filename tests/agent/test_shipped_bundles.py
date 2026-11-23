"""Every bundle in assistant_models/ was trained or benchmarked on this app's tools.

The GUI no longer tells users about a schema mismatch (they cannot act on it), so
this is where a tool change catches a stale model: benchmark the model on the new
tools and add the hash to its manifest's ``verified_schemas``, or retrain it.
"""
import pytest

pytest.importorskip('pydantic_ai')

from frictionsim2d.agent.bundles import MODELS_DIR, catalogue  # noqa: E402
from frictionsim2d.agent.runtime import tool_schema_hash  # noqa: E402

SHIPPED = {name: tune for name, tune in catalogue(MODELS_DIR).items() if tune.bundle}


def test_there_are_shipped_bundles():
    assert SHIPPED


@pytest.mark.parametrize('name', sorted(SHIPPED))
def test_shipped_bundle_fits_the_current_tools(name):
    tune = SHIPPED[name]
    assert tune.fits(tool_schema_hash()), (
        f'{name} was trained on tool schema {tune.schema_hash} and verified on '
        f'{list(tune.verified_schemas)}, not on the current {tool_schema_hash()}')
