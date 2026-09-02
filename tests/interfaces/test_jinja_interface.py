from unittest.mock import patch

import pytest
from jinja2 import TemplateNotFound

from src.interfaces.jinja import PackageLoader


def test_get_source_missing_template_raises_template_not_found():
    loader = PackageLoader('src.templates')
    with pytest.raises(TemplateNotFound):
        loader.get_source(environment=None, template='does/not/exist.lmp')


def test_get_source_does_not_mask_type_errors():
    loader = PackageLoader('src.templates')
    with patch.object(loader, '_package', None):
        with pytest.raises(Exception) as excinfo:
            loader.get_source(environment=None, template='afm/slide.lmp')
        assert not isinstance(excinfo.value, TemplateNotFound)
