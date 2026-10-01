"""Nynxz Save nodes — Save Image plus the Save Info hub that configures it.

`Save Info` holds the folder/format/metadata config on one panel; any number of
`Save Image` nodes wire to it and each contributes only its own short name. Importing
`api` here is what registers the /nynxz/* HTTP routes the panel calls.

Node classes are found by discovery from the modules that define them, so this file
deliberately does NOT re-export them.
"""

from . import api  # noqa: F401  (registers /nynxz/* routes on import)
