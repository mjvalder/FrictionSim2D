"""The analysis contract and its registry.

An :class:`Analysis` turns the integrity-pass points of one dataset (a
:class:`~frictionsim2d.analysis.source.RunSource`) into one table. Each subclass is
registered by name with :func:`register`; :func:`get_analysis` looks it up.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from importlib import import_module
from typing import TYPE_CHECKING, Any, ClassVar, Dict, FrozenSet, Literal, Optional, Tuple, Type

from pydantic import BaseModel

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

    from .source import RunSource

__all__ = ['ANALYSES', 'IDENTITY_COLUMNS', 'Analysis', 'check_identity', 'get_analysis',
           'identity_columns', 'register']

Level = Literal['run', 'run_state', 'material']

#: The columns every table of a level carries.
IDENTITY_COLUMNS: Dict[str, Tuple[str, ...]] = {
    'run': ('run_uuid', 'material', 'layer', 'load', 'load_kind', 'angle'),
    'run_state': ('run_uuid', 'material', 'layer', 'load', 'load_kind', 'angle'),
    'material': ('material', 'layer'),
}


class Analysis(ABC):
    """One on-demand analysis of a dataset.

    Subclasses set the class attributes and implement :meth:`compute`.
    ``version`` is bumped whenever the table an unchanged input gives changes:
    a stored result is reused only under the same name, version, parameters
    and dataset.
    """

    name: ClassVar[str]
    version: ClassVar[str]
    level: ClassVar[Level]
    families: ClassVar[FrozenSet[str]]
    Params: ClassVar[Type[BaseModel]]
    #: The columns every row of the table carries (:func:`identity_columns`);
    #: None: the level's :data:`IDENTITY_COLUMNS`.
    identity: ClassVar[Optional[Tuple[str, ...]]] = None

    @abstractmethod
    def compute(self, source: 'RunSource', params: BaseModel) -> 'pd.DataFrame':
        """The table of ``source`` under ``params`` (an instance of ``Params``)."""

    @classmethod
    def depends_on(cls, dataset: str,  # pylint: disable=unused-argument
                   params: BaseModel) -> Dict[str, Any]:  # pylint: disable=unused-argument
        """The other stored results this analysis reads over ``dataset``
        under ``params`` (an instance of ``Params``), as ``name -> result
        node`` (None: not stored). They are part of a result's identity (a
        stored result is reused only while they are the same) and are linked
        as inputs of the result. None by default."""
        return {}


def identity_columns(cls: Type[Analysis]) -> Tuple[str, ...]:
    """The columns a table of ``cls`` must carry: its own ``identity``, else
    its level's :data:`IDENTITY_COLUMNS`."""
    return cls.identity if cls.identity is not None else IDENTITY_COLUMNS[cls.level]


def check_identity(cls: Type[Analysis], table: 'pd.DataFrame') -> None:
    """Raise ``ValueError`` naming the identity columns (:func:`identity_columns`)
    a non-empty ``table`` of ``cls`` lacks."""
    missing = [c for c in identity_columns(cls) if c not in table.columns]
    if len(table) and missing:
        raise ValueError(f'{cls.name}: its table lacks the identity columns {missing}')


#: name -> registered analysis class.
ANALYSES: Dict[str, Type[Analysis]] = {}


def register(cls: Type[Analysis]) -> Type[Analysis]:
    """Class decorator: add ``cls`` to :data:`ANALYSES` under ``cls.name``.

    Raises:
        ValueError: another analysis already has that name.
    """
    name = cls.name
    if name in ANALYSES and ANALYSES[name] is not cls:
        raise ValueError(f'an analysis named {name!r} is already registered '
                         f'({ANALYSES[name].__qualname__})')
    ANALYSES[name] = cls
    return cls


def get_analysis(name: str) -> Type[Analysis]:
    """The registered analysis ``name``.

    Raises:
        KeyError: no analysis of that name.
    """
    import_module('.catalogue', __package__)  # registers every analysis
    try:
        return ANALYSES[name]
    except KeyError:
        raise KeyError(f'no analysis named {name!r} (known: {sorted(ANALYSES)})') from None
