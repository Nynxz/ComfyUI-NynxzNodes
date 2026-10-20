#!/usr/bin/env python3
"""Run every check script that needs no ComfyUI checkout, and say which failed.

Run: python scripts/run_checks.py            (from the repo root; CI runs this too)

They need torch, numpy and Pillow (CPU builds are fine). The checks that drive ComfyUI itself —
check_moss_regions, check_nodes, check_qwen21_contract — take a checkout path and are run by hand:

    python scripts/check_moss_regions.py /path/to/ComfyUI
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent

#: Need a ComfyUI checkout (they import comfy), so they don't run here.
NEED_COMFYUI = {"check_moss_regions.py", "check_nodes.py", "check_qwen21_contract.py"}


def main() -> int:
    scripts = sorted(p for p in SCRIPTS.glob("check_*.py") if p.name not in NEED_COMFYUI)
    failed = []
    for script in scripts:
        start = time.perf_counter()
        result = subprocess.run(
            [sys.executable, str(script)], capture_output=True, text=True, cwd=SCRIPTS.parent
        )
        took = time.perf_counter() - start
        ok = result.returncode == 0
        print(f"{'ok  ' if ok else 'FAIL'} {script.name} ({took:.1f}s)")
        if not ok:
            failed.append(script.name)
            print(result.stdout[-4000:] + result.stderr[-4000:])
    print(f"\n{len(scripts) - len(failed)}/{len(scripts)} check scripts passed.")
    if NEED_COMFYUI:
        print("Not run here (need a ComfyUI checkout): " + ", ".join(sorted(NEED_COMFYUI)))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
