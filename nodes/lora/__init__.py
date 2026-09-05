"""Nynxz LoRA nodes — the on-node searchable/bookmarkable stack loader.

Importing `api` here is what registers the /nynxz/* HTTP routes the widget calls.
Node classes are found by discovery from the modules that define them, so this
file deliberately does NOT re-export them.
"""

from . import api  # noqa: F401  (registers /nynxz/* routes on import)
