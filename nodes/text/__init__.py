"""Saved Text — pick a kept piece of text (a system prompt, a style line) on the node.

Importing `api` here is what registers the /nynxz/texts routes the widget calls. Node classes are
found by discovery from the modules that define them, so this file does not re-export them.
"""

from . import api  # noqa: F401  (registers /nynxz/texts routes on import)
