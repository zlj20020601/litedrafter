#!/usr/bin/env python3
"""E0 diag: find running==0 valleys (point boundaries) in raw gauge."""
import json
import gzip

BASE = "/root/autodl-tmp/litedrafter/outputs"

for name in ["ar", "dflash"]:
    with gzip.open(f"{BASE}/c_eff_{name}_raw_20260816.json.gz", "rt") as f:
        raw = json.load(f)
    print(f"\n=== {name.upper()} n={len(raw)} span={raw[-1]['t']-raw[0]['t']:.0f}s ===")
    # contiguous valleys of running==0 (waiting may be >0?)
    valleys = []
    i = 0
    while i < len(raw):
        if raw[i]["g"]["running"] == 0 and raw[i]["g"]["waiting"] == 0:
            j = i
            while j < len(raw) and raw[j]["g"]["running"] == 0 and raw[j]["g"]["waiting"] == 0:
                j += 1
            dur = raw[j-1]["t"] - raw[i]["t"]
            valleys.append((i, j, dur))
            i = j
        else:
            i += 1
    print(f"valleys (running=0&waiting=0): {len(valleys)}")
    for i0, i1, dur in valleys:
        t_rel = raw[i0]["t"] - raw[0]["t"]
        # context: running before/after
        rb = raw[max(0, i0-1)]["g"]["running"]
        ra = raw[min(len(raw)-1, i1)]["g"]["running"]
        print(f"  idx {i0:>6}-{i1:>6}  t+{t_rel:>7.0f}s  dur={dur:>6.1f}s  run_before={rb:.0f} run_after={ra:.0f}")
