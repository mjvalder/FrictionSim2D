"""Writing configuration files.

:func:`src.core.config.parse_config` is the reader half of a small format
registry (``.ini`` / ``.yaml`` / ``.json``); this module is the writer half.
Anything that authors a config -- currently the GUI -- goes through here so that
what it produces is exactly what ``parse_config`` reads back.

The ``.ini`` writer is the inverse of :func:`src.core.utils.read_config`, whose
value grammar is implicit and easy to violate:

* an empty right-hand side reads back as ``None``, so ``None`` must be written
  as an empty value, never as the string ``"None"``;
* list detection is triggered by a trailing ``]``, so lists are written as JSON;
* ``value.isdigit()`` decides int-vs-float, so a float must carry a decimal
  point or an exponent or it will read back as an ``int``. Note that
  ``"-5".isdigit()`` is ``False``, so a *negative* integer reads back as a
  float. That asymmetry is pre-existing behaviour of the reader; the writer
  reproduces whatever it was given and the round-trip test pins it;
* everything after a ``#`` is stripped as an inline comment, so a value
  containing ``#`` cannot survive and is rejected here rather than silently
  truncated on the next read.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

import yaml

__all__ = ['write_config', 'format_ini_value', 'SUPPORTED_WRITE_FORMATS']

SUPPORTED_WRITE_FORMATS = ('.ini', '.yaml', '.yml', '.json')

#: Mapping of ``section -> key -> comment`` used to annotate ``.ini`` output.
CommentMap = Mapping[str, Mapping[str, str]]


def format_ini_value(value: Any) -> str:
    """Render a Python value as an ``.ini`` right-hand side.

    Inverse of the value parsing in :func:`src.core.utils.read_config`.

    Raises:
        ValueError: If the value contains ``#``, which the reader would strip.
    """
    if value is None:
        return ''

    if isinstance(value, bool):
        # The reader has no bool rule; these come back as strings and Pydantic
        # coerces them. 'True'/'False' is what the existing configs use.
        return 'True' if value else 'False'

    if isinstance(value, (list, tuple)):
        rendered = json.dumps(list(value))
    elif isinstance(value, float):
        rendered = repr(value)
        # repr() of an integral float already includes '.0'; guard anything odd
        # (e.g. a float subclass) that would read back as an int.
        if rendered.isdigit():
            rendered += '.0'
    elif isinstance(value, int):
        rendered = str(value)
    else:
        rendered = str(value)

    if '#' in rendered:
        raise ValueError(
            f"Config values cannot contain '#': {rendered!r}. The reader treats "
            "it as the start of an inline comment and would truncate the value."
        )
    return rendered


def _render_ini(config: Mapping[str, Mapping[str, Any]],
                comments: Optional[CommentMap] = None,
                header: Optional[str] = None) -> str:
    """Render a section mapping as ``.ini`` text.

    Written by hand rather than via ``configparser.write`` so that inline
    comments can be attached and so that ``%`` needs no interpolation escaping.
    """
    comments = comments or {}
    lines = []

    if header:
        lines.extend(f"# {line}" if line else "#" for line in header.splitlines())
        lines.append("")

    for section, values in config.items():
        lines.append(f"[{section}]")
        section_comments = comments.get(section, {})
        for key, value in values.items():
            rendered = format_ini_value(value)
            comment = section_comments.get(key)
            if comment:
                lines.append(f"{key} = {rendered} # {comment}")
            else:
                lines.append(f"{key} = {rendered}")
        lines.append("")

    return "\n".join(lines).rstrip("\n") + "\n"


def write_config(config: Mapping[str, Any],
                 path: Union[str, Path],
                 *,
                 comments: Optional[CommentMap] = None,
                 header: Optional[str] = None) -> Path:
    """Write a config mapping to ``path``, dispatching on the file extension.

    Args:
        config: Section mapping, in the shape ``parse_config`` returns.
        path: Destination. The suffix selects the format.
        comments: Optional ``section -> key -> comment`` annotations. Applied to
            ``.ini`` output only; other formats ignore it.
        header: Optional comment block for the top of an ``.ini`` file.

    Returns:
        The path written.

    Raises:
        ValueError: If the extension is not supported.
    """
    path = Path(path)
    ext = path.suffix.lower()

    if ext == '.ini':
        text = _render_ini(config, comments=comments, header=header)
    elif ext in ('.yaml', '.yml'):
        text = yaml.safe_dump(_to_plain(config), sort_keys=False)
    elif ext == '.json':
        text = json.dumps(_to_plain(config), indent=2) + "\n"
    else:
        raise ValueError(
            f"Unsupported configuration file format: {ext}. "
            f"Supported formats: {', '.join(SUPPORTED_WRITE_FORMATS)}"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')
    return path


def _to_plain(value: Any) -> Any:
    """Convert nested mappings to plain dicts for the yaml/json writers."""
    if isinstance(value, Mapping):
        return {k: _to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(v) for v in value]
    return value


def round_trip(config: Mapping[str, Any], path: Union[str, Path]) -> Dict[str, Any]:
    """Write ``config`` then read it back. Used by tests and by the GUI's
    save-and-verify path."""
    from .config import parse_config  # noqa: PLC0415  (circular at module level)

    written = write_config(config, path)
    return parse_config(written)
