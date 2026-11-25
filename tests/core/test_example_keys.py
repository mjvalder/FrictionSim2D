"""Every key an example sets must be one the config model reads.

The config models ignore unknown keys, so a key that is misspelt or in the wrong
section validates cleanly and does nothing. ``finite_sheet_afm_edge_mode`` in
``[general]`` is the case this guards: it is a ``settings.yaml``
(``geometry``) key, so the run quietly kept free edges. An example teaches
users what to write, so an ignored key in one gets copied.
"""
from __future__ import annotations

import re
from typing import List, Type

import pytest
from pydantic import BaseModel

from frictionsim2d.core.config import (
    AFMSimulationConfig,
    SheetOnSheetSimulationConfig,
    parse_config,
)
from tests.helpers.examples import discover, friction_examples

#: ``run`` models whose root config is AFMSimulationConfig.
_AFM_MODELS = {'afm', 'indent', 'pes_tip'}
#: Numbered sections are folded into a list of this field's item model.
_NUMBERED = {re.compile(r'^2D-\d+$'): 'sheet', re.compile(r'^Flake-\d+$'): 'sheet'}
#: Sections whose keys are free-form by design.
_FREE_FORM = {'lj_override'}


def _root_model(model: str) -> Type[BaseModel]:
    return AFMSimulationConfig if model in _AFM_MODELS else SheetOnSheetSimulationConfig


def _names(model: Type[BaseModel]) -> dict:
    """``{name or alias: field}`` for every field of ``model``."""
    out = {}
    for name, field in model.model_fields.items():
        out[name] = field
        if field.alias:
            out[field.alias] = field
    return out


def _sub_model(field) -> Type[BaseModel]:
    annotation = field.annotation
    for candidate in (annotation, *getattr(annotation, '__args__', ())):
        if isinstance(candidate, type) and issubclass(candidate, BaseModel):
            return candidate
    raise TypeError(f"{field} is not a model-valued field")


def ignored_keys(raw: dict, root: Type[BaseModel]) -> List[str]:
    """``[section] key`` entries in ``raw`` that ``root`` would silently drop."""
    root_fields = _names(root)
    found = []
    for section, values in raw.items():
        if section in _FREE_FORM or not isinstance(values, dict):
            continue
        field_name = next((f for pattern, f in _NUMBERED.items() if pattern.match(section)),
                          section)
        if field_name not in root_fields:
            found.append(f"[{section}] (the whole section)")
            continue
        accepted = _names(_sub_model(root_fields[field_name]))
        found += [f"[{section}] {key}" for key in values if key not in accepted]
    return found


@pytest.mark.parametrize('example', friction_examples(), ids=lambda e: e.rel)
def test_every_example_key_is_a_field_the_model_reads(example):
    root = _root_model(example.model)
    ignored = ignored_keys(parse_config(example.path), root)
    assert not ignored, f"{example.rel}: ignored by {root.__name__}: {ignored}"


def test_a_settings_key_in_general_is_reported(tmp_path):
    (tmp_path / 'afm').mkdir()
    (tmp_path / 'afm' / 'x.ini').write_text(
        "[general]\ntemp = 300\nfinite_sheet_afm_edge_mode = spring\n", encoding='utf-8')
    [example] = discover(tmp_path)
    assert ignored_keys(parse_config(example.path), AFMSimulationConfig) == [
        "[general] finite_sheet_afm_edge_mode"]


def test_a_section_the_model_does_not_have_is_reported():
    assert ignored_keys({'tip': {'r': 10}}, SheetOnSheetSimulationConfig) == [
        "[tip] (the whole section)"]
