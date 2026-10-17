"""Layouts — boxes on a canvas, shared by every editor that lays a picture out.

The editor itself lives in the frontend (frontend/layout/): one canvas with undo, a right-click
menu, copy and paste, and JSON in and out, whatever the boxes mean. A mode says what they mean —
Fresco's regions, or a model's own layout prompt.

    library.py   saved layouts on disk, one JSON file each (pure, no server)
    api.py       the /nynxz/layouts routes over it

Importing `api` here is what registers the routes.
"""

from . import api  # noqa: F401  (registers /nynxz/layouts routes on import)
