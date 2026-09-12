#!/usr/bin/env python3
"""E0 diag2: sustained valleys (>=5s) + running+waiting step detection."""
import json
import gzip

BASE = "/root/autodl-tmp/litedrafter/outputs"
EXPECT_C = [1, 4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 64, 80, 96]

for name in ["ar", "dflash"]:
    with gzip.open(f"{BASE}/c_eff_{name}_raw_20260816.json.gz", "rt") as f:
        raw = json.load(f)
    print(f"\n=== {name.upper()} span={raw[-1]['t']-raw[0]['t']:.0f}s ===")
    # sustained valleys >= 5s
    valleys = []
    i = 0
    while i < len(raw):
        if raw[i]["g"]["running"] == 0 and raw[i]["g"]["waiting"] == 0:
            j = i
            while j < len(raw) and raw[j]["g"]["running"] == 0 and raw[j]["g"]["waiting"] == 0:
                j += 1
            dur = raw[j-1]["t"] - raw[i]["t"]
            if dur >= 5.0:
                valleys.append((i, j, dur))
            i = j
        else:
            i += 1
    print(f"sustained valleys >=5s: {len(valleys)}")
    for i0, i1, dur in valleys:
        t_rel = raw[i0]["t"] - raw[0]["t"]
        print(f"  idx {i0:>6}-{i1:>6}  t+{t_rel:>7.0f}s  dur={dur:>6.1f}s")
    # rolling median of running+waiting (window ~ 15s)
    W = max(7, int(15 / 0.21))
    med = []
    vals = [s["g"]["running"] + s["g"]["waiting"] for s in raw]
    import statistics as st
    for k in range(len(vals)):
        lo = max(0, k - W // 2); hi = min(len(vals), k + W // 2)
        med.append(st.median(vals[lo:hi]))
    # step boundaries: where med crosses from near one expected level to another
    steps = []
    k = 0
    for k in range(1, len(med)):
        d = med[k] - med[k-1]
        if abs(d) >= 3:
            steps.append((k, d, raw[k]["t"] - raw[0]["t"]))
    # merge consecutive steps within 30s
    merged = []
    for k, d, tr in steps:
        if merged and tr - merged[-1][2] < 30:
            merged[-1] = (k, d, tr)
        else:
            merged.append((k, d, tr))
    print(f"steps |delta|>=3 in 15s-median: {len(merged)}")
    for k, d, tr in merged:
        print(f"  idx {k:>6}  t+{tr:>7.0f}s  delta={d:+.0f}  -> med_before={med[max(0,k-5)]:.0f} med_after={med[min(len(med)-1,k+5)]:.0f}")
