#!/usr/bin/env python3
"""Run every check script, and say which failed.

Run: python scripts/run_checks.py                    (from the repo root; CI runs this too)
     python scripts/run_checks.py /path/to/ComfyUI   (also the checks that drive ComfyUI itself)

They need torch, numpy and Pillow (CPU builds are fine). The checks that import comfy take the
checkout as their argument and run with whatever Python this is, so give that checkout's Python
when passing one. They run on CPU whatever the machine has.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent

#: Need a ComfyUI checkout (they import comfy): run only when one is given.
NEED_COMFYUI = {
    "check_moss_regions.py",
    "check_nodes.py",
    "check_qwen21_contract.py",
    "check_region_lora.py",
}


def main(comfyui: str | None) -> int:
    scripts = sorted(p for p in SCRIPTS.glob("check_*.py") if comfyui or p.name not in NEED_COMFYUI)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    failed = []
    for script in scripts:
        args = [sys.executable, str(script)]
        if script.name in NEED_COMFYUI:
            args.append(comfyui)
        start = time.perf_counter()
        result = subprocess.run(args, capture_output=True, text=True, cwd=SCRIPTS.parent, env=env)
        took = time.perf_counter() - start
        ok = result.returncode == 0
        print(f"{'ok  ' if ok else 'FAIL'} {script.name} ({took:.1f}s)")
        if not ok:
            failed.append(script.name)
            print(result.stdout[-4000:] + result.stderr[-4000:])
    print(f"\n{len(scripts) - len(failed)}/{len(scripts)} check scripts passed.")
    if not comfyui:
        print("Not run here (need a ComfyUI checkout): " + ", ".join(sorted(NEED_COMFYUI)))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else None))
