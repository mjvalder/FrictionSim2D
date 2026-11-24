"""No deck template may read a key its builder did not pass.

The production environments use jinja2's default ``Undefined``, which renders a
renamed or forgotten key as an empty field and fails nothing. Rebuilding every
snapshot case under ``StrictUndefined`` turns that silence into an error; the
``| default(...)`` and ``is defined`` forms the templates use on purpose still
pass.
"""
import jinja2
import pytest

from tests.helpers.deck_snapshot import render, render_set

_CASES = render_set()


@pytest.mark.parametrize("case", _CASES, ids=[c.name for c in _CASES])
def test_every_template_key_is_passed(case, lammps_replay, tmp_path, monkeypatch):
    original = jinja2.Environment.__init__

    def strict_init(self, *args, **kwargs):
        kwargs["undefined"] = jinja2.StrictUndefined
        original(self, *args, **kwargs)

    monkeypatch.setattr(jinja2.Environment, "__init__", strict_init)
    render(case, tmp_path)
