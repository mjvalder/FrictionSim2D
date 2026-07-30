"""Qt widgets. Everything here imports PySide6."""

from .fields import FieldEditor, build_editor
from .form import SectionForm, ConfigForm

# Imported for the side effect of registering their editors with
# fields.EDITOR_REGISTRY. fields.py cannot import them itself -- they import it.
from . import bands, keyvalue, materials  # noqa: F401

__all__ = ['FieldEditor', 'build_editor', 'SectionForm', 'ConfigForm']
