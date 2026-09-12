#!/usr/bin/env python3
"""E0 step 2: summary JSON structure (segment/C-point metadata)."""
import json

BASE = "/root/autodl-tmp/litedrafter/outputs"

for name in ["ar", "dflash"]:
    p = f"{BASE}/c_eff_{name}_20260816.json"
    print(f"\n{'='*60}\n=== {name} summary ===\n{'='*60}")
    with open(p) as f:
        d = json.load(f)
    print("top-level:", type(d).__name__)
    if isinstance(d, dict):
        for k, v in d.items():
            if isinstance(v, list):
                print(f"  {k}: list len={len(v)}")
                if v and isinstance(v[0], dict):
                    print(f"    [0] keys: {list(v[0].keys())}")
                    print(f"    [0]: {json.dumps(v[0], ensure_ascii=False)[:400]}")
            elif isinstance(v, dict):
                print(f"  {k}: dict keys={list(v.keys())[:10]}")
            else:
                print(f"  {k}: {str(v)[:100]}")
    elif isinstance(d, list):
        print(f"list len={len(d)}")
        if d:
            print("[0]:", json.dumps(d[0], ensure_ascii=False)[:500])
