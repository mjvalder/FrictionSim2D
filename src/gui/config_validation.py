"""Validating a raw config mapping and reporting failures per field.

Constructing a simulation config can fail in two different ways, and only one
of them is a ``ValidationError``:

* ordinary field problems (wrong type, out of range, cross-field rules) raise
  ``pydantic.ValidationError``, which carries a ``loc`` we can map to a section
  and key;
* a missing potential or CIF file raises ``FileNotFoundError`` from inside the
  ``pot_path`` / ``cif_path`` field validator. Pydantic v2 only converts
  ``ValueError`` and ``AssertionError``, so this propagates uncaught -- and it
  is the most common authoring mistake there is. Four of the nine configs in
  ``examples/`` fail this way.

:func:`validate_config` turns both into one uniform report, so a caller can
decorate the offending field instead of surfacing a traceback.

Model semantics are untouched: the CLI still gets the raise. This is a wrapper
around construction, not a change to the models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Type, Union

from pydantic import BaseModel, ValidationError

from ..run.placeholders import TEMPLATE_MARKERS, has_template_marker
from ..core.utils.paths import _find_existing  # noqa: PLC2701

__all__ = ['ConfigIssue', 'ConfigValidation', 'validate_config', 'preflight_paths']

#: Config keys that name a file on disk, and the resolver each one needs.
PATH_KEYS = ('pot_path', 'cif_path')

#: Issue kinds raised by the path preflight. Used to suppress the same problem
#: arriving a second time, unlocated, from inside a field validator.
PATH_ISSUE_KINDS = ('missing_file', 'unexpanded_template')


@dataclass(frozen=True)
class ConfigIssue:
    """One problem with a config, located as precisely as possible."""

    message: str
    section: Optional[str] = None
    key: Optional[str] = None
    kind: str = 'validation'
    #: Position within a list-valued section, when the failing value sits in
    #: one -- ``sheets[1].cif_path`` is ``section='sheets'``, ``index=1``,
    #: ``key='cif_path'``. Section and key alone cannot say which member
    #: failed, and the key must stay the field name or there is nothing for a
    #: form to decorate.
    index: Optional[int] = None

    @property
    def location(self) -> str:
        section = self.section
        if section and self.index is not None:
            section = f"{section}[{self.index}]"
        if section and self.key:
            return f"[{section}] {self.key}"
        if section:
            return f"[{section}]"
        return "(config)"

    def render(self) -> str:
        return f"{self.location}: {self.message}"


@dataclass
class ConfigValidation:
    """Outcome of validating a config mapping."""

    issues: List[ConfigIssue] = field(default_factory=list)
    config: Optional[BaseModel] = None

    @property
    def ok(self) -> bool:
        return not self.issues

    def issues_for(self, section: str, key: Optional[str] = None) -> List[ConfigIssue]:
        """Issues attached to a section, or to one key within it."""
        return [
            i for i in self.issues
            if i.section == section and (key is None or i.key == key)
        ]

    def render(self) -> str:
        if self.ok:
            return "config is valid"
        return "\n".join(i.render() for i in self.issues)


def preflight_paths(config: Mapping[str, Any],
                    base_dir: Optional[Union[str, Path]] = None) -> List[ConfigIssue]:
    """Check every file-valued key before the model is constructed.

    Doing this first is what makes a missing potential land on the right field:
    the exception raised later from inside the validator does not say which
    section it came from.

    Unexpanded ``{mat}``/``{mat1}``/``{mat2}`` templates are reported as their
    own kind, because they are not user error -- they are configs that must
    be expanded before they can be validated at all. Missing this for a
    heterostructure pair's ``{mat1}``/``{mat2}`` (rather than just ``{mat}``)
    would classify it as a missing file instead -- a hard error that
    permanently blocks Render/Generate, since a template placeholder can
    never resolve to an existing path.
    """
    issues: List[ConfigIssue] = []
    for section, values in config.items():
        if not isinstance(values, Mapping):
            continue
        for key in PATH_KEYS:
            value = values.get(key)
            if not isinstance(value, str) or not value:
                continue
            if has_template_marker(value):
                markers = '/'.join(TEMPLATE_MARKERS)
                issues.append(ConfigIssue(
                    message=(
                        f"'{value}' still contains an unexpanded template "
                        f"placeholder ({markers}); expand the material sweep "
                        "before validating"
                    ),
                    section=section, key=key, kind='unexpanded_template',
                ))
                continue
            if _find_existing(value, base_dir) is None:
                searched = f" (searched relative to {base_dir})" if base_dir else ""
                issues.append(ConfigIssue(
                    message=f"file not found: '{value}'{searched}",
                    section=section, key=key, kind='missing_file',
                ))
    return issues


def _issue_from_pydantic_error(error: Mapping[str, Any]) -> ConfigIssue:
    """Map one pydantic error dict onto a section/key.

    A loc through a list -- ``('sheets', 1, 'cif_path')``, which is how every
    second-sheet failure of a heterostructure config arrives -- keeps the
    index separately and the *field* as the key. Taking the first two elements
    would report ``key='1'``, naming no field, and a form that decorates by
    key would then decorate nothing however precisely it maps the section.
    """
    loc: Sequence[Any] = error.get('loc', ())
    section: Optional[str] = None
    key: Optional[str] = None
    index: Optional[int] = None
    if len(loc) >= 3 and isinstance(loc[1], int):
        section, index, key = str(loc[0]), loc[1], str(loc[2])
    elif len(loc) >= 2:
        section, key = str(loc[0]), str(loc[1])
    elif len(loc) == 1:
        section = str(loc[0])
    return ConfigIssue(
        message=error.get('msg', 'invalid value'),
        section=section, key=key, kind='validation', index=index,
    )


def validate_config(config: Mapping[str, Any],
                    model: Type[BaseModel],
                    *,
                    base_dir: Optional[Union[str, Path]] = None,
                    skip_preflight: bool = False) -> ConfigValidation:
    """Validate ``config`` against ``model``, reporting issues per field.

    Args:
        config: Raw config mapping, in the shape ``parse_config`` returns, with
            a ``settings`` entry already present.
        model: The root config model to construct.
        base_dir: Directory that relative ``pot_path`` / ``cif_path`` values are
            resolved against. Passed to the model through the validation
            context as well, so a successful validation resolves the same way.
        skip_preflight: Skip the file-existence pass. Only useful when the
            caller has already run it.

    Returns:
        A :class:`ConfigValidation`. ``ok`` is True only when the model was
        constructed and no issues were found.
    """
    issues: List[ConfigIssue] = []
    if not skip_preflight:
        issues.extend(preflight_paths(config, base_dir))

    context: Dict[str, Any] = {}
    if base_dir is not None:
        context['base_dir'] = str(base_dir)

    try:
        instance = model.model_validate(dict(config), context=context or None)
    except ValidationError as exc:
        issues.extend(_issue_from_pydantic_error(e) for e in exc.errors())
        return ConfigValidation(issues=issues)
    except (FileNotFoundError, OSError) as exc:
        # A path validator raised. The preflight above will normally already
        # have located it -- as a missing file, or as a path still holding an
        # unexpanded {mat} -- in which case this is the same problem arriving
        # again without a location, and adding it would just be noise. Only
        # report it when the preflight found nothing.
        if not any(i.kind in PATH_ISSUE_KINDS for i in issues):
            issues.append(ConfigIssue(message=str(exc), kind='missing_file'))
        return ConfigValidation(issues=issues)
    except (ValueError, TypeError) as exc:
        issues.append(ConfigIssue(message=str(exc), kind='error'))
        return ConfigValidation(issues=issues)

    return ConfigValidation(issues=issues, config=instance)
