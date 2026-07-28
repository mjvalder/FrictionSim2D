"""Shared styling.

Qt's ``palette(mid)`` resolves to something close to the background on a dark
theme, which made the explanatory captions -- the whole point of which is to be
read -- effectively invisible. These are explicit values chosen to stay legible
on both a light and a dark background.
"""

from __future__ import annotations

__all__ = ['CAPTION', 'CAPTION_SMALL', 'SECTION_HELP', 'WARNING', 'REQUIRED_MARK']

#: Secondary text: readable on either background, clearly not the primary text.
_SECONDARY = '#9aa0a6'

CAPTION = f'color: {_SECONDARY};'
CAPTION_SMALL = f'color: {_SECONDARY}; font-size: 11px;'
SECTION_HELP = f'color: {_SECONDARY}; font-style: italic;'
WARNING = 'color: #e8a33d;'

#: Marks a field the model requires.
REQUIRED_MARK = ' <b>*</b>'
