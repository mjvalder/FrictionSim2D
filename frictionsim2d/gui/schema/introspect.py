"""Turning pydantic config models into a description a form can be built from.

Reads the models at run time so a field added to, say, ``SubstrateConfig``
appears in the GUI with no GUI edit -- the property that keeps this package from
conflicting with config-model work happening elsewhere.

Two root-model fields are deliberately excluded:

``settings``
    A whole ``GlobalSettings`` tree. It belongs to the Settings tab, and walking
    into it here would bury the run config under nine unrelated sub-models.

``sheets``
    Derived, not authored: a model validator folds ``[2D-1]``/``[2D-2]``
    sections into it, and two or more entries route the build to the
    heterostructure builders. It has no ``.ini`` section of its own, so
    rendering a control for it would offer something that cannot be written.

Standard library and pydantic only: no Qt.
"""

from __future__ import annotations

import enum
import typing
from dataclasses import dataclass, field, replace
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple, Type, Union, get_args, get_origin

from pydantic import BaseModel
from pydantic_core import PydanticUndefined

__all__ = [
    'FieldKind', 'FieldSpec', 'SectionSpec',
    'describe_model', 'describe_section', 'EXCLUDED_ROOT_FIELDS',
]

#: Root-model fields that are not part of the authored config. See module docs.
EXCLUDED_ROOT_FIELDS = frozenset({'settings', 'sheets'})

#: Field names whose string value names a file.
_PATH_SUFFIXES = ('_path',)


class FieldKind(enum.Enum):
    """What sort of control a field needs."""

    BOOL = 'bool'
    INT = 'int'
    FLOAT = 'float'
    STR = 'str'
    PATH = 'path'
    ENUM = 'enum'
    LIST = 'list'
    DICT = 'dict'
    #: A value that may be one number or a list of them, where the arity
    #: changes the meaning -- a single load, or a set to loop over.
    SCALAR_OR_LIST = 'scalar_or_list'
    NESTED = 'nested'
    UNKNOWN = 'unknown'


_NoneType = type(None)


@dataclass(frozen=True)
class FieldSpec:
    """One editable field."""

    name: str
    key: str
    kind: FieldKind
    required: bool
    default: Any = None
    doc: str = ''
    choices: Tuple[Any, ...] = ()
    item_kind: Optional[FieldKind] = None
    annotation: Any = None


@dataclass
class SectionSpec:
    """One config section, i.e. one ``[section]`` in an ``.ini``."""

    name: str
    key: str
    model: Type[BaseModel]
    fields: List[FieldSpec] = field(default_factory=list)
    optional: bool = False
    doc: str = ''

    def field_map(self) -> Dict[str, FieldSpec]:
        return {f.key: f for f in self.fields}


def _unwrap_optional(annotation: Any) -> Tuple[Any, bool]:
    """Strip ``Optional`` and report whether it was there."""
    if get_origin(annotation) is Union:
        args = [a for a in get_args(annotation) if a is not _NoneType]
        if len(args) < len(get_args(annotation)):
            return (args[0] if len(args) == 1 else Union[tuple(args)]), True
    return annotation, False


def _is_model(annotation: Any) -> bool:
    return isinstance(annotation, type) and issubclass(annotation, BaseModel)


def _scalar_kind(annotation: Any, name: str) -> FieldKind:
    if annotation is bool:
        return FieldKind.BOOL
    if annotation is int:
        return FieldKind.INT
    if annotation is float:
        return FieldKind.FLOAT
    if annotation is str:
        return FieldKind.PATH if name.endswith(_PATH_SUFFIXES) else FieldKind.STR
    return FieldKind.UNKNOWN


def _classify(annotation: Any, name: str) -> Tuple[FieldKind, Tuple[Any, ...], Optional[FieldKind]]:
    """Kind, choices and item kind for an annotation."""
    annotation, _ = _unwrap_optional(annotation)
    origin = get_origin(annotation)

    if origin is typing.Literal:
        return FieldKind.ENUM, get_args(annotation), None

    if origin in (list, List):
        args = get_args(annotation)
        item = _scalar_kind(args[0], name) if args else FieldKind.UNKNOWN
        return FieldKind.LIST, (), item

    if origin in (dict, Dict):
        # The value type, not the key type -- KeyValueGridEditor needs it to
        # give a newly added row the right control (a checkbox for
        # Dict[str, bool], a text cell otherwise) without having to guess
        # from a runtime value that may just be the empty string add_row()
        # is handed by default.
        args = get_args(annotation)
        item = _scalar_kind(args[1], name) if len(args) > 1 else FieldKind.UNKNOWN
        return FieldKind.DICT, (), item

    if origin is Union:
        args = get_args(annotation)
        scalars = [a for a in args if get_origin(a) not in (list, List)]
        lists = [a for a in args if get_origin(a) in (list, List)]
        # float | list[float]: one value or a set to loop over.
        if scalars and lists:
            return FieldKind.SCALAR_OR_LIST, (), _scalar_kind(scalars[0], name)
        if scalars:
            return _scalar_kind(scalars[0], name), (), None
        return FieldKind.UNKNOWN, (), None

    if _is_model(annotation):
        return FieldKind.NESTED, (), None

    if isinstance(annotation, type) and issubclass(annotation, enum.Enum):
        return FieldKind.ENUM, tuple(m.value for m in annotation), None

    return _scalar_kind(annotation, name), (), None


def _default_of(info: Any) -> Any:
    if info.default is not PydanticUndefined:
        return info.default
    if info.default_factory is not None:
        try:
            return info.default_factory()
        except TypeError:
            return None
    return None


def describe_section(model: Type[BaseModel], name: str, key: str,
                     optional: bool = False) -> SectionSpec:
    """Describe one config section's model."""
    spec = SectionSpec(
        name=name, key=key, model=model, optional=optional,
        doc=(model.__doc__ or '').strip().split('\n')[0],
    )

    for field_name, info in model.model_fields.items():
        kind, choices, item_kind = _classify(info.annotation, field_name)
        spec.fields.append(FieldSpec(
            name=field_name,
            key=info.alias or field_name,
            kind=kind,
            required=info.is_required(),
            default=_default_of(info),
            doc=(info.description or '').strip(),
            choices=choices,
            item_kind=item_kind,
            annotation=info.annotation,
        ))

    return spec


def describe_model(model: Type[BaseModel]) -> List[SectionSpec]:
    """Describe a root config model as a list of sections.

    Each nested model field becomes one section; ``settings`` and ``sheets`` are
    excluded (see module docs). A ``Dict`` field is also a section, because that
    is how it is written in an ``.ini``: the section's keys are the dict's.

    Cached on the model class, which is the whole of the input: a class's
    fields are fixed once it is defined, so two calls with the same class
    cannot disagree. The cache matters because
    ``ConfigDocument.conflicts()`` -- run on every Render and every Generate
    -- reaches one field default through ``section()``, and each of those
    walked every model of the config.

    A model *switch* cannot serve a stale answer: the two families have
    different root classes, so they are different cache keys. The one thing
    that would break it is a class edited in place after being described,
    which nothing does outside a debugger; ``cache_clear`` is there if it
    ever becomes necessary.

    Each call still gets its own ``SectionSpec`` objects and its own field
    lists, because callers edit what they are handed --
    ``ConfigDocument.second_material_spec`` drops ``materials_list`` from the
    list it gets. ``FieldSpec`` is frozen, so those are shared as they are.
    """
    return [replace(spec, fields=list(spec.fields))
            for spec in _describe_model(model)]


@lru_cache(maxsize=None)
def _describe_model(model: Type[BaseModel]) -> Tuple[SectionSpec, ...]:
    """The walk itself. See `describe_model`, which owns the copying."""
    sections: List[SectionSpec] = []

    for field_name, info in model.model_fields.items():
        if field_name in EXCLUDED_ROOT_FIELDS:
            continue

        annotation, is_optional = _unwrap_optional(info.annotation)
        section_key = info.alias or field_name

        if _is_model(annotation):
            sections.append(describe_section(
                annotation, name=field_name, key=section_key,
                optional=is_optional or not info.is_required(),
            ))
        elif get_origin(annotation) in (dict, Dict):
            # A free-form section: keys are chosen by the user, not the schema.
            sections.append(SectionSpec(
                name=field_name, key=section_key, model=model, optional=True,
                doc='Free-form key/value overrides.',
            ))

    return tuple(sections)


#: Exposed on the public name, which is the one callers know. Nothing in the
#: app calls it -- a described class never changes -- but a test proving the
#: walk happens at all has to be able to empty the cache first.
describe_model.cache_clear = _describe_model.cache_clear
