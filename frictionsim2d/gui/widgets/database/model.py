"""The results table's model, proxy and column vocabulary.

The first `QAbstractTableModel` in this codebase, and deliberately so: every
other table here is a `QTableWidget`, which has to be rebuilt cell by cell to
sort and cannot sort by anything but its own display text. A store with
thousands of results needs the opposite -- format for the eye, sort and filter
on the value.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PySide6.QtCore import (QAbstractTableModel, QModelIndex, QObject, Qt,
                             QSortFilterProxyModel)
from PySide6.QtGui import QBrush, QColor

from ...style import STATUS_ERROR_COLOR, STATUS_OK_COLOR, STATUS_WARN_COLOR

__all__ = ['CURATED_COLUMNS', 'Column', 'LOAD_UNITS', 'ResultSortProxyModel',
           'ResultTableModel', 'SORT_ROLE', 'STATUS_LEVELS', 'columns_for_record',
           'display_value', 'join_list', 'record_value', 'sort_value',
           'status_level']

#: The role a view sorts on: the raw value, not the formatted text. '30.00 nN'
#: and '9.00 nN' sort the wrong way round as strings.
SORT_ROLE = Qt.UserRole + 1

#: What a load is measured in, decided per record by `load_kind`. A tip is
#: pushed with a force, a sheet is pressed with a pressure, and one column
#: shows both.
LOAD_UNITS: Dict[str, str] = {'force': 'nN', 'pressure': 'GPa'}

#: Shown where a record has no value at all -- 'None' reads like a stored
#: string, and an empty cell reads like a bug.
MISSING = '—'

#: Every verdict status (`run_status.STATUSES`) and how alarming it is. A
#: `quarantined` point is unjudged rather than broken, so it warns; every
#: other non-complete status is a point whose numbers must not be used.
STATUS_LEVELS: Dict[str, str] = {
    'complete': 'ok', 'quarantined': 'warn',
    'truncated': 'error', 'no_output': 'error', 'unreadable': 'error',
    'halted': 'error', 'failed_integrity': 'error', 'not_reached': 'error',
}

_LEVEL_COLORS = {'warn': STATUS_WARN_COLOR, 'error': STATUS_ERROR_COLOR,
                 'ok': STATUS_OK_COLOR}


def status_level(record: Dict[str, Any]) -> str:
    """'ok', 'warn' or 'error' for one row, from its verdict `status`.

    A point with no verdict (`status` None: never checked) warns -- its
    numbers are not known to be usable. A status outside the vocabulary
    warns too, rather than passing as 'ok'.
    """
    status = str(record.get('status') or '').lower()
    if not status:
        return 'warn'
    return STATUS_LEVELS.get(status, 'warn')


@dataclass(frozen=True)
class Column:
    """One table column: where the value comes from and how it reads."""

    key: str
    title: str
    fmt: str = '{}'
    unit: str = ''
    align_right: bool = False
    #: Keys to try when `key` is absent.
    fallbacks: Tuple[str, ...] = ()
    #: How a bool reads in this column.
    truthy_text: Tuple[str, str] = ('yes', 'no')


#: The curated subset, over a `Friction2DDB.query` row: the run tokens, the
#: point's swept axes, the selected summary's COF and the selected verdict.
CURATED_COLUMNS: Tuple[Column, ...] = (
    Column('set_label', 'Set'),
    Column('family', 'Family'),
    Column('materials', 'Materials'),
    Column('layers', 'Layers', align_right=True),
    Column('load', 'Load', fmt='{:.2f}', align_right=True),
    Column('angle', 'Angle', fmt='{:.1f}', unit='°', align_right=True),
    Column('speed', 'Speed', fmt='{:.2f}', unit='Å/ps', align_right=True),
    Column('temperature', 'T', fmt='{:.0f}', unit='K', align_right=True),
    Column('mean_cof', 'mean COF', fmt='{:.4f}', align_right=True),
    Column('std_cof', 'std COF', fmt='{:.4f}', align_right=True),
    Column('status', 'Status'),
    Column('reasons', 'Reasons'),
)

_CURATED_KEYS = tuple(c.key for c in CURATED_COLUMNS)


def join_list(value: Sequence[Any]) -> str:
    """Render a list/tuple field as one '+'-joined string.

    Shared by `display_value` and `sort_value` -- both need the same
    stand-in for a list, one to show it and one to sort/filter by it.
    """
    return '+'.join(str(v) for v in value)


def record_value(record: Dict[str, Any], column: Column) -> Any:
    """The column's value in this record, or None."""
    for key in (column.key, *column.fallbacks):
        value = record.get(key)
        if value is not None:
            return value
    return None


def sort_value(record: Dict[str, Any], column: Column) -> Any:
    """A comparable stand-in for the value.

    Qt compares whatever `SORT_ROLE` returns; a list or a None in one row and
    a float in another raises inside the proxy's comparison.
    """
    value = record_value(record, column)
    if value is None:
        return ''
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (list, tuple)):
        return join_list(value)
    if isinstance(value, dict):
        return str(value)
    return value


def _unit(record: Dict[str, Any], column: Column) -> str:
    """The unit to print after the value.

    A load's unit is decided by the row's `load_kind`: getting this wrong
    prints a pressure in nN.
    """
    if column.unit:
        return column.unit
    if column.key != 'load':
        return ''
    return LOAD_UNITS.get(str(record.get('load_kind') or ''), '')


def display_value(record: Dict[str, Any], column: Column) -> str:
    """What the cell reads: formatted, with its unit, '—' when absent."""
    value = record_value(record, column)
    if value is None:
        return MISSING
    if isinstance(value, bool):
        text = column.truthy_text[0] if value else column.truthy_text[1]
    elif isinstance(value, (list, tuple)):
        text = join_list(value)
    else:
        try:
            text = column.fmt.format(value)
        except (ValueError, TypeError):
            text = str(value)

    unit = _unit(record, column)
    return f"{text} {unit}".strip()


def columns_for_record(record: Dict[str, Any]) -> Tuple[Column, ...]:
    """Every field of a record as a column, curated ones first.

    The 'All columns' toggle: the curated subset answers "which run is this",
    the rest answers "what else was stored", and the order keeps the first
    question answerable without scrolling.
    """
    if not record:
        return CURATED_COLUMNS
    extra = tuple(Column(key, key) for key in record
                  if key not in _CURATED_KEYS)
    curated = tuple(c for c in CURATED_COLUMNS
                    if c.key in record or any(f in record for f in c.fallbacks))
    return curated + extra


class ResultTableModel(QAbstractTableModel):
    """A list of result records, viewed through a column spec."""

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._records: List[Dict[str, Any]] = []
        self._columns: Tuple[Column, ...] = CURATED_COLUMNS
        self._show_all = False

    # ------------------------------------------------------------- contents

    def set_records(self, records: Sequence[Dict[str, Any]]) -> None:
        self.beginResetModel()
        self._records = [dict(r) for r in records]
        self._columns = self._pick_columns()
        self.endResetModel()

    def set_all_columns(self, show_all: bool) -> None:
        self.beginResetModel()
        self._show_all = bool(show_all)
        self._columns = self._pick_columns()
        self.endResetModel()

    def _pick_columns(self) -> Tuple[Column, ...]:
        if not self._show_all or not self._records:
            return CURATED_COLUMNS
        return columns_for_record(self._records[0])

    @property
    def columns(self) -> Tuple[Column, ...]:
        return self._columns

    def records(self) -> List[Dict[str, Any]]:
        return list(self._records)

    def record_at(self, row: int) -> Dict[str, Any]:
        return self._records[row]

    # ------------------------------------------------------- QAbstractModel

    def rowCount(self, parent=QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else len(self._records)

    def columnCount(self, parent=QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else len(self._columns)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        record = self._records[index.row()]
        column = self._columns[index.column()]
        if role == Qt.DisplayRole:
            return display_value(record, column)
        if role == SORT_ROLE:
            return sort_value(record, column)
        if role == Qt.TextAlignmentRole:
            return int(Qt.AlignRight | Qt.AlignVCenter) if column.align_right \
                else int(Qt.AlignLeft | Qt.AlignVCenter)
        if role == Qt.ForegroundRole:
            level = status_level(record)
            # 'ok' is left to the palette: painting every good row green
            # would make the colour meaningless and fight both themes.
            return QBrush(QColor(_LEVEL_COLORS[level])) if level != 'ok' else None
        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):  # noqa: N802
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            return self._columns[section].title
        return section + 1


class ResultSortProxyModel(QSortFilterProxyModel):
    """Sorts on the raw value; narrows the fetched rows by free text.

    The free-text box filters here rather than in the worker on purpose:
    typing must not re-query the store, and it must not change the cached
    query's identity (D13). Every field is searched, including the ones no
    column shows -- a uuid pasted from a log finds its row.
    """

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self.setSortRole(SORT_ROLE)
        self._text = ''

    def set_text_filter(self, text: str) -> None:
        self._text = (text or '').strip().lower()
        self.invalidateRowsFilter()

    def filterAcceptsRow(self, source_row: int, source_parent) -> bool:  # noqa: N802
        del source_parent
        if not self._text:
            return True
        model = self.sourceModel()
        if model is None:
            return True
        record = model.record_at(source_row)
        return any(self._text in str(value).lower()
                   for value in record.values() if value is not None)
