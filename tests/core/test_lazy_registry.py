import pytest

from frictionsim2d import core


def test_unknown_attribute_raises_attribute_error_not_import_error():
    with pytest.raises(AttributeError, match="has no attribute 'nope'"):
        core.nope  # noqa: B018  (attribute access is the test)
