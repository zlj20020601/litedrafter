#!/usr/bin/env python3
"""E0 step 1: recon structure of c_eff raw gauge files (read-only)."""
import json
import gzip

BASE = "/root/autodl-tmp/litedrafter/outputs"

def describe(obj, depth=0, max_depth=4):
    pad = "  " * depth
    if depth > max_depth:
        return
    if isinstance(obj, dict):
        for k in list(obj.keys())[:12]:
            v = obj[k]
            if isinstance(v, (dict, list)):
                print(f"{pad}{k}: {type(v).__name__} len={len(v)}")
                describe(v, depth + 1, max_depth)
            else:
                print(f"{pad}{k}: {type(v).__name__} = {str(v)[:80]}")
    elif isinstance(obj, list) and obj:
        print(f"{pad}[0] of {len(obj)}:")
        describe(obj[0], depth + 1, max_depth)

for name in ["ar", "dflash"]:
    p = f"{BASE}/c_eff_{name}_raw_20260816.json.gz"
    print(f"\n{'='*60}\n=== {name} ===\n{'='*60}")
    with gzip.open(p, "rt") as f:
        d = json.load(f)
    print("top-level:", type(d).__name__)
    describe(d)
