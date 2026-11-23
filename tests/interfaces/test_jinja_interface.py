from unittest.mock import patch

import pytest
from jinja2 import TemplateNotFound

from frictionsim2d.interfaces.jinja import PackageLoader


def test_get_source_missing_template_raises_template_not_found():
    loader = PackageLoader('frictionsim2d.templates')
    with pytest.raises(TemplateNotFound):
        loader.get_source(environment=None, template='does/not/exist.lmp')


def test_get_source_does_not_mask_type_errors():
    loader = PackageLoader('frictionsim2d.templates')
    with patch.object(loader, '_package', None):
        with pytest.raises(Exception) as excinfo:
            loader.get_source(environment=None, template='afm/slide.lmp')
        assert not isinstance(excinfo.value, TemplateNotFound)


from pathlib import Path

from jinja2 import Environment


def test_get_source_returns_text_name_and_uptodate_for_a_real_template():
    loader = PackageLoader('frictionsim2d.templates')
    source, name, uptodate = loader.get_source(Environment(), 'common/observables.lmp')
    assert 'macro' in source and name == 'common/observables.lmp' and uptodate() is True


def test_get_source_maps_a_vanishing_file_to_template_not_found(monkeypatch):
    loader = PackageLoader('frictionsim2d.templates')
    def vanish(self, *a, **k):
        raise FileNotFoundError(str(self))
    monkeypatch.setattr(Path, 'read_text', vanish)
    with pytest.raises(TemplateNotFound):
        loader.get_source(Environment(), 'common/observables.lmp')
