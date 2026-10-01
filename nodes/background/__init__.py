"""Nynxz Background Maker — packages a resolution-independent background spec for the grid.

No HTTP routes: the panel's pattern list and its swatch previews are both client-side
(frontend/lib/bgPatterns.ts), mirroring PATTERNS in config.py.

Node classes are found by discovery from the modules that define them, so this file
deliberately does NOT re-export them.
"""
