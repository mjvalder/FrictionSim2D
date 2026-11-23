"""Everything the store holds about one selected row."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from PySide6.QtWidgets import (
    QHeaderView,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .model import join_list

__all__ = ['DetailPanel', 'format_bytes']

_UNITS = (('MB', 1024 * 1024), ('kB', 1024))


def format_bytes(count: int) -> str:
    """'812 B', '39.3 kB', '5.0 MB'."""
    for suffix, size in _UNITS:
        if count >= size:
            return f"{count / size:.1f} {suffix}"
    return f"{int(count)} B"


class DetailPanel(QWidget):
    """One point: its attributes, the verdict and summary the policy selects,
    the `related` counts, and the stored file list.

    Groups rather than one flat list: the same key can mean different things
    on two nodes, and the file list is the answer to a different question
    ("is it worth downloading") than the attributes are.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.tree = QTreeWidget(self)
        self.tree.setColumnCount(2)
        self.tree.setHeaderLabels(['Field', 'Value'])
        self.tree.header().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.tree.setRootIsDecorated(True)
        layout.addWidget(self.tree)

    # --------------------------------------------------------------- filling

    def show_detail(self, detail: Dict[str, Any]) -> None:
        """Render one worker `detail` response, replacing what is shown."""
        self.tree.clear()

        for title, key in (('Point', 'point'), ('Verdict', 'verdict'),
                           ('Summary', 'summary')):
            node = detail.get(key)
            if node is None:
                self._group(title, [(f'{key} uuid', 'none')])
                continue
            self._group(title, [(f'{key} uuid', str(node.get('uuid') or ''))]
                        + _attribute_rows(node.get('attributes')))

        related = detail.get('related') or {}
        if related:
            self._group('Related (differ only along)',
                        [(kind, str(count)) for kind, count in related.items()])

        files = detail.get('files') or []
        if files:
            rows = [(f"{f['node']}: {f['path']}", format_bytes(int(f['bytes'])))
                    for f in files]
            rows.append(('total stored',
                         format_bytes(int(detail.get('total_bytes') or 0))))
            self._group('Files', rows)

        self.tree.expandAll()

    def _group(self, title: str, rows: List[Tuple[str, str]]) -> None:
        parent = QTreeWidgetItem(self.tree, [title, ''])
        for key, value in rows:
            QTreeWidgetItem(parent, [key, value])

    def clear(self) -> None:
        self.tree.clear()

    # ----------------------------------------------------------------- reads

    def rows(self) -> List[Tuple[str, str]]:
        """Every visible (label, value) pair, groups flattened away."""
        out: List[Tuple[str, str]] = []
        for index in range(self.tree.topLevelItemCount()):
            group = self.tree.topLevelItem(index)
            for child in range(group.childCount()):
                item = group.child(child)
                out.append((item.text(0), item.text(1)))
        return out


def _attribute_rows(attributes: Optional[Dict[str, Any]]) -> List[Tuple[str, str]]:
    """Sorted key/value pairs; lists join, everything else is str()."""
    rows: List[Tuple[str, str]] = []
    for key in sorted(attributes or {}):
        value = attributes[key]
        if isinstance(value, (list, tuple)):
            text = join_list(value)
        else:
            text = str(value)
        rows.append((key, text))
    return rows
