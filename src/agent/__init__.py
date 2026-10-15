"""The GUI assistant's headless core.

Settings, the path sandbox, change sets, tools and knowledge need only the
base install. `providers` and `runtime` need the optional ``agent`` extra
(pydantic-ai, Python >= 3.11): check `_AGENT_AVAILABLE` before importing them.
Nothing here imports Qt.
"""
from importlib.util import find_spec

_AGENT_AVAILABLE = find_spec('pydantic_ai') is not None

__all__ = ['_AGENT_AVAILABLE']
