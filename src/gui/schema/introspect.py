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
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type, Union, get_args, get_origin

from pydantic import BaseModel

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
    constraints: Dict[str, Any] = field(default_factory=dict)
    annotation: Any = None

    @property
    def is_pathlike(self) -> bool:
        return self.kind is FieldKind.PATH


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
        args = [a for a in get_args(annotation) if a is not type(None)]
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
        return FieldKind.DICT, (), None

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


def _constraints(info: Any) -> Dict[str, Any]:
    """Numeric and length bounds declared on a field."""
    out: Dict[str, Any] = {}
    for meta in getattr(info, 'metadata', ()) or ():
        for attribute in ('ge', 'gt', 'le', 'lt', 'min_length', 'max_length'):
            value = getattr(meta, attribute, None)
            if value is not None:
                out[attribute] = value
    return out


def _default_of(info: Any) -> Any:
    from pydantic_core import PydanticUndefined  # noqa: PLC0415

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
            constraints=_constraints(info),
            annotation=info.annotation,
        ))

    return spec


def describe_model(model: Type[BaseModel]) -> List[SectionSpec]:
    """Describe a root config model as a list of sections.

    Each nested model field becomes one section; ``settings`` and ``sheets`` are
    excluded (see module docs). A ``Dict`` field is also a section, because that
    is how it is written in an ``.ini``: the section's keys are the dict's.
    """
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

    return sections
