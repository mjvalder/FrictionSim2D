"""A fake ``lmp``: write the files a LAMMPS deck would write, without running it.

It does not interpret LAMMPS. It reads a deck line by line and acts on four
commands only:

- ``variable NAME index|string|equal|loop ...`` -- recorded, so file names expand;
- ``fix ID GROUP ave/time NEVERY NREPEAT NFREQ q1 q2 ... file PATTERN [title1 "..."]``
  -- the file gets the two ave/time header lines (``title1`` if given, else
  LAMMPS' default) and ``rows`` deterministic rows, one column per listed
  ``v_``/``c_``/``f_`` quantity;
- ``print "TEXT" file|append PATH`` -- the text, once; a line that carries an
  immediate ``$(...)`` it cannot evaluate is a data row and is written ``rows``
  times with deterministic numbers;
- ``write_data PATH`` -- a copy of the deck's most recent ``read_data`` input
  (so an end snapshot equals the start structure), else a header-only file.

The deck is walked once per element of the cartesian product of its ``index``
variables' values; ``${NAME}``/``$NAME``/``$(v_NAME)`` expand as LAMMPS would
(``index``/``string`` verbatim, ``equal`` and immediates as ``%.15g``; an
``equal`` it cannot evaluate warns and expands to ``0``, and ``**`` is never
evaluated). Commands
inside an ``if`` are never run, so a halt registry is never written. Finally it
writes a log (``log.lammps`` in ``cwd``, or ``-log``) with a ``Loop time`` and a
``Total wall time`` line.

    python tests/scenarios/fake_lmp.py -in DECK [-log FILE] [-rows N]
"""
from __future__ import annotations

import argparse
import itertools
import re
import shlex
import shutil
import sys
import warnings
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

#: What the fake log claims to be; `aiida.calcjob` records the version string.
VERSION = 'fake_lmp'

_AVE_TIME_KEYWORDS = frozenset({
    'mode', 'file', 'ave', 'start', 'off', 'overwrite', 'format',
    'title1', 'title2', 'title3', 'append'})
_PRINT_KEYWORDS = frozenset({'file', 'append', 'screen', 'universe'})
_IMMEDIATE = re.compile(r'\$\(([^()]*(?:\([^()]*\)[^()]*)*)\)')
_BRACED = re.compile(r'\$\{(\w+)\}')
_BARE = re.compile(r'\$([A-Za-z0-9])')
_VREF = re.compile(r'\bv_(\w+)')
_ARITHMETIC = re.compile(r'^[\d.eE+\-*/()\s]+$')


class _Unknown(ValueError):
    """An expression this fake cannot evaluate."""


def _logical_lines(text: str) -> Iterator[str]:
    """Deck lines with ``&`` continuations joined."""
    pending = ''
    for raw in text.splitlines():
        line = raw.rstrip()
        if line.endswith('&'):
            pending += line[:-1] + ' '
            continue
        yield pending + line
        pending = ''
    if pending:
        yield pending


def _strip_comment(line: str) -> str:
    """``line`` up to its first ``#`` outside quotes."""
    quote = None
    for i, char in enumerate(line):
        if quote:
            if char == quote:
                quote = None
        elif char in '"\'':
            quote = char
        elif char == '#':
            return line[:i]
    return line


def _option(args: Sequence[str], key: str) -> Optional[str]:
    """The argument after keyword ``key`` in ``args``, or None."""
    for i, arg in enumerate(args[:-1]):
        if arg == key:
            return args[i + 1]
    return None


def _fmt(value: float) -> str:
    return f'{value:.15g}'


class _Deck:
    """One walk of a deck under one assignment of its index variables."""

    def __init__(self, cwd: Path, rows: int, assignment: Dict[str, str]):
        self.cwd = cwd
        self.rows = rows
        self.assignment = assignment
        self.vars: Dict[str, Tuple[str, str]] = {}
        self.last_input: Optional[Path] = None
        self.written: List[Path] = []
        self.counter = 0

    # -------------------------------------------------------------- values
    def number(self, name: str) -> float:
        """The numeric value of variable ``name``."""
        if name not in self.vars:
            raise _Unknown(name)
        style, value = self.vars[name]
        if style == 'equal':
            return self.evaluate(value)
        try:
            return float(value)
        except ValueError as exc:
            raise _Unknown(name) from exc

    def evaluate(self, expr: str) -> float:
        """A constant arithmetic expression over known ``v_`` references."""
        expr = _VREF.sub(lambda m: repr(self.number(m.group(1))), expr.strip())
        # `**` refused: a constant like 9**9**9 would hang eval.
        if not _ARITHMETIC.match(expr) or '**' in expr:
            raise _Unknown(expr)
        try:
            return float(eval(expr, {'__builtins__': {}}, {}))  # pylint: disable=eval-used
        except (SyntaxError, ZeroDivisionError, TypeError) as exc:
            raise _Unknown(expr) from exc

    def text(self, name: str) -> str:
        """``${name}``: index/string/loop verbatim, equal as ``%.15g``."""
        style, value = self.vars.get(name, ('string', '0'))
        if style == 'equal':
            try:
                return _fmt(self.evaluate(value))
            except _Unknown:
                warnings.warn(f'fake_lmp: equal-style ${{{name}}} = {value!r} cannot be '
                              "evaluated; substituted '0'", stacklevel=2)
                return '0'
        return value

    def fake_number(self) -> str:
        """A deterministic stand-in for a value only a real run knows."""
        self.counter += 1
        return f'{1.0 + 0.01 * self.counter:.6g}'

    def immediate(self, match: re.Match) -> str:
        expr, _, fmt = match.group(1).partition(':')
        try:
            value = self.evaluate(expr)
        except _Unknown:
            return self.fake_number()
        return (fmt % value) if fmt else _fmt(value)

    def substitute(self, line: str) -> str:
        """Expand ``$(...)``, ``${NAME}`` and ``$N`` outside single quotes."""
        parts = re.split(r"('[^']*')", line)
        for i in range(0, len(parts), 2):
            part = _IMMEDIATE.sub(self.immediate, parts[i])
            part = _BRACED.sub(lambda m: self.text(m.group(1)), part)
            parts[i] = _BARE.sub(lambda m: self.text(m.group(1)), part)
        return ''.join(parts)

    # ------------------------------------------------------------ commands
    def path(self, token: str) -> Path:
        path = Path(token)
        return path if path.is_absolute() else self.cwd / path

    def write(self, path: Path, text: str, append: bool = False) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'a' if append else 'w', encoding='utf-8') as handle:
            handle.write(text)
        if path not in self.written:
            self.written.append(path)

    def variable(self, args: Sequence[str]) -> None:
        if len(args) < 2:
            return
        name, style, values = args[0], args[1], list(args[2:])
        if style == 'delete':
            self.vars.pop(name, None)
        elif style == 'index' and values:
            self.vars.setdefault(name, ('index', self.assignment.get(name, values[0])))
        elif style == 'loop' and values:
            self.vars.setdefault(name, ('loop', '1'))
        elif style == 'string' and values:
            self.vars.setdefault(name, ('string', values[0]))
        elif style == 'equal' and values:
            self.vars[name] = ('equal', ' '.join(values))

    def ave_time(self, args: Sequence[str]) -> None:
        """``fix ID GROUP ave/time NEVERY NREPEAT NFREQ values... keywords...``."""
        fix_id, nfreq = args[0], int(float(args[5]))
        quantities: List[str] = []
        i = 6
        while i < len(args) and args[i] not in _AVE_TIME_KEYWORDS:
            if args[i][:2] in ('v_', 'c_', 'f_'):
                quantities.append(args[i])
            i += 1
        target = _option(args[i:], 'file')
        if target is None:
            return
        title1 = _option(args[i:], 'title1') or f'# Time-averaged data for fix {fix_id}'
        title2 = _option(args[i:], 'title2') or '# TimeStep ' + ' '.join(quantities)
        body = [title1, title2]
        for row in range(self.rows):
            values = ' '.join(f'{1.0 + 0.1 * row + 0.01 * col:.6g}'
                              for col in range(len(quantities)))
            body.append(f'{(row + 1) * nfreq} {values}')
        self.write(self.path(target), '\n'.join(body) + '\n')

    def print_(self, raw: str) -> None:
        """``print "TEXT" ... file|append PATH``; a data row is repeated."""
        args = shlex.split(raw)[1:]
        split = next((i for i, a in enumerate(args) if a in _PRINT_KEYWORDS), len(args))
        mode = next((k for k in ('file', 'append') if _option(args[split:], k)), None)
        if mode is None:
            return
        template = ' '.join(args[:split])
        is_row = mode == 'append' and _IMMEDIATE.search(template) is not None
        lines = []
        for _ in range(self.rows if is_row else 1):
            lines.append(self.substitute(template))
        path = self.path(self.substitute(_option(args[split:], mode)))
        self.write(path, '\n'.join(lines) + '\n', append=mode == 'append')

    def write_data(self, token: str) -> None:
        path = self.path(token)
        if self.last_input is not None and self.last_input.is_file():
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(self.last_input, path)
            if path not in self.written:
                self.written.append(path)
        else:
            self.write(path, 'LAMMPS data file via fake_lmp\n\n0 atoms\n')

    def run(self, deck_text: str) -> None:
        for logical in _logical_lines(deck_text):
            line = _strip_comment(logical).strip()
            if not line:
                continue
            command = line.split(None, 1)[0]
            if command == 'print':
                self.print_(line)
                continue
            if command not in ('variable', 'fix', 'read_data', 'write_data'):
                continue
            args = shlex.split(self.substitute(line))[1:]
            if command == 'variable':
                self.variable(args)
            elif command == 'fix' and len(args) >= 6 and args[2] == 'ave/time':
                self.ave_time(args)
            elif command == 'read_data' and args:
                self.last_input = self.path(args[0])
            elif command == 'write_data' and args:
                self.write_data(args[0])


def _index_values(deck_text: str) -> Dict[str, List[str]]:
    """``{name: values}`` of every ``variable NAME index ...``, first definition wins."""
    found: Dict[str, List[str]] = {}
    for logical in _logical_lines(deck_text):
        args = _strip_comment(logical).split()
        if len(args) > 3 and args[0] == 'variable' and args[2] == 'index':
            found.setdefault(args[1], args[3:])
    return found


def _log_text(rows: int) -> str:
    return (f'LAMMPS ({VERSION})\n'
            f'Loop time of 1 on 1 procs for {rows} steps with 0 atoms\n'
            'Total wall time: 0:00:01\n')


def fake_run(deck: Path, rows: int = 5, *, cwd: Optional[Path] = None,
             log: Optional[Path] = None) -> List[Path]:
    """Write every file ``deck`` would write; return their paths.

    Relative paths resolve against ``cwd`` (default: the current directory),
    as ``lmp -in DECK`` resolves them; the log goes to ``log`` (default
    ``cwd/log.lammps``), as ``-log`` does.
    """
    deck = Path(deck)
    cwd = Path.cwd() if cwd is None else Path(cwd)
    text = deck.read_text(encoding='utf-8')
    index = _index_values(text)
    names = sorted(index)
    written: List[Path] = []
    for combo in itertools.product(*(index[n] for n in names)):
        walk = _Deck(cwd, rows, dict(zip(names, combo)))
        walk.run(text)
        written.extend(p for p in walk.written if p not in written)
    log_path = cwd / 'log.lammps' if log is None else Path(log)
    if not log_path.is_absolute():
        log_path = cwd / log_path
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(_log_text(rows), encoding='utf-8')
    written.append(log_path)
    return written


def main(argv: Optional[Sequence[str]] = None) -> int:
    """``fake_lmp.py -in DECK [-log FILE] [-rows N]``, in the current directory."""
    parser = argparse.ArgumentParser(prog='fake_lmp', description=__doc__.splitlines()[0])
    parser.add_argument('-in', dest='deck', required=True, type=Path)
    parser.add_argument('-log', dest='log', type=Path, default=None)
    parser.add_argument('-rows', dest='rows', type=int, default=5)
    args = parser.parse_args(argv)
    for path in fake_run(args.deck, args.rows, log=args.log):
        print(path)
    return 0


if __name__ == '__main__':
    sys.exit(main())
