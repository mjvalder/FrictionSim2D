"""Shared styling.

Qt's ``palette(mid)`` resolves to something close to the background on a dark
theme, which made the explanatory captions -- the whole point of which is to be
read -- effectively invisible. These are explicit values chosen to stay legible
on both a light and a dark background.
"""

from __future__ import annotations

__all__ = ['CAPTION', 'CAPTION_SMALL', 'SECTION_HELP', 'WARNING', 'REQUIRED_MARK',
           'STATUS_OK', 'STATUS_WARN', 'STATUS_ERROR', 'STATUS_OK_COLOR',
           'STATUS_WARN_COLOR', 'STATUS_ERROR_COLOR']

#: Secondary text: readable on either background, clearly not the primary text.
_SECONDARY = '#9aa0a6'

CAPTION = f'color: {_SECONDARY};'
CAPTION_SMALL = f'color: {_SECONDARY}; font-size: 11px;'
SECTION_HELP = f'color: {_SECONDARY}; font-style: italic;'
WARNING = 'color: #e8a33d;'

#: Marks a field the model requires.
REQUIRED_MARK = ' <b>*</b>'

#: Run status. The bare colours exist because QColor cannot take a stylesheet
#: string; the stylesheet tokens are built from them, so each status is still
#: one hex value in one place.
STATUS_OK_COLOR = '#4caf50'
STATUS_WARN_COLOR = '#e8a33d'
STATUS_ERROR_COLOR = '#e05c5c'

STATUS_OK = f'color: {STATUS_OK_COLOR};'
STATUS_WARN = f'color: {STATUS_WARN_COLOR};'
STATUS_ERROR = f'color: {STATUS_ERROR_COLOR};'
