"""Plain-function tools the assistant calls. Each takes `AgentDeps` first.

Tools never write: a tool that wants a change returns a handle and leaves a
change set in `deps.pending` for the user to apply. The helpers here are shared
by the tools and the change sets.
"""
import uuid

#: Upper bound on a tool result, in characters. Small-model context is the
#: bottleneck, so every result is cut to this by `compact`.
RESULT_LIMIT = 1500
#: Most characters a change handle (a tool result) may take.
HANDLE_LIMIT = 600


def compact(text: str, limit: int = RESULT_LIMIT) -> str:
    """Cut ``text`` to ``limit`` characters on a line boundary, saying how much went."""
    if len(text) <= limit:
        return text
    lines = text.splitlines()
    if len(lines[0]) + 1 > limit - 40:  # nothing fits on a line boundary
        return text[:limit - 20] + ' ... (truncated)'
    kept, size = [], 0
    for index, line in enumerate(lines):
        if size + len(line) + 1 > limit - 40:
            kept.append(f'... {len(lines) - index} more lines')
            break
        kept.append(line)
        size += len(line) + 1
    return '\n'.join(kept)


def capped(head: str, instruction: str) -> str:
    """``head`` cut so the whole handle fits `HANDLE_LIMIT`; ``instruction`` kept whole."""
    return compact(head, HANDLE_LIMIT - len(instruction) - 1) + ' ' + instruction


def clean_error(exc: Exception) -> str:
    """An exception's text without pydantic's "For further information visit" lines."""
    lines = [line.strip() for line in str(exc).splitlines() if line.strip()]
    return '\n'.join(line for line in lines
                     if not line.startswith('For further information'))


def new_id() -> str:
    """A short random change-set id."""
    return uuid.uuid4().hex[:6]
