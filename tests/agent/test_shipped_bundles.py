"""Every published fine-tune was trained or benchmarked on this app's tools.

The tool schemas each model fits are recorded in `bundles.PUBLISHED_SCHEMAS`
(from its manifest on Hugging Face), so this is where a tool change catches a
stale model: benchmark the model on the new tools and add the hash to its
manifest's ``verified_schemas`` and to `PUBLISHED_SCHEMAS`, or retrain it.
"""
import pytest

pytest.importorskip('pydantic_ai')

from frictionsim2d.agent.bundles import PUBLISHED, PUBLISHED_SCHEMAS  # noqa: E402
from frictionsim2d.agent.runtime import tool_schema_hash  # noqa: E402


def test_every_published_model_records_its_schemas():
    assert PUBLISHED and set(PUBLISHED_SCHEMAS) == set(PUBLISHED)
    assert all(PUBLISHED_SCHEMAS.values())


@pytest.mark.parametrize('tag', sorted(PUBLISHED))
def test_shipped_bundle_fits_the_current_tools(tag):
    assert tool_schema_hash() in PUBLISHED_SCHEMAS[tag], (
        f'fs2d-assistant:{tag} was trained or verified on tool schemas '
        f'{list(PUBLISHED_SCHEMAS[tag])}, not on the current {tool_schema_hash()}')
