"""Plain-function tools the assistant calls. Each takes `AgentDeps` first.

Tools never write: a tool that wants a change returns a handle and leaves a
change set in `deps.pending` for the user to apply.
"""

#: Upper bound on a tool result, in characters. Small-model context is the
#: bottleneck (spec §3), so every result is cut to this by `compact`.
RESULT_LIMIT = 1500


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
