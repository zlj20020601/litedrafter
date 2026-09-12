#!/usr/bin/env python3
"""E0: steady-state re-analysis of 0816 raw gauge data.

Goals (per 0817 design doc):
1. Per C-point steady-state stats: running/waiting/kv_usage mean/p50/p95/max
   on the SETTLED segment (exclude initial burst), plus waiting>0 time fraction.
2. Check whether max values only come from initial burst.
3. Record whether scheduled_tokens was captured (gauge schema says no).
4. Point-set inventory: C values scanned, gaps vs E2 targets.

Method: segment raw time series into activity bursts (running>0 or waiting>0),
merge on gaps > 60s of inactivity, match segments in order to summary points.
Steady state = last 80% of each segment (settling exclusion), plus a
"plateau mode" check (modal running value & fraction at mode).
"""
import json
import gzip
import statistics as st

BASE = "/root/autodl-tmp/litedrafter/outputs"
GAP_S = 60.0          # inactivity gap that splits segments
STEADY_FRAC = 0.8     # steady state = final 80% of segment duration

def load_raw(name):
    with gzip.open(f"{BASE}/c_eff_{name}_raw_20260816.json.gz", "rt") as f:
        return json.load(f)

def segment(samples):
    """split on inactivity gaps -> list of (i_start, i_end) index slices"""
    segs = []
    start = None
    for i, s in enumerate(samples):
        active = s["g"]["running"] > 0 or s["g"]["waiting"] > 0
        if active and start is None:
            start = i
        elif not active and start is not None:
            # check gap length before closing
            if i + 1 < len(samples):
                segs.append((start, i))
                start = None
            else:
                segs.append((start, i + 1))
                start = None
    if start is not None:
        segs.append((start, len(samples)))
    # merge segments separated by short gaps (< GAP_S): walk again on times
    merged = []
    for s0, s1 in segs:
        if merged and (samples[s0]["t"] - samples[merged[-1][1] - 1]["t"]) < GAP_S:
            merged[-1] = (merged[-1][0], s1)
        else:
            merged.append((s0, s1))
    return merged

def pctl(vals, q):
    vs = sorted(vals)
    if not vs:
        return float("nan")
    k = (len(vs) - 1) * q
    f = int(k)
    c = min(f + 1, len(vs) - 1)
    return vs[f] + (vs[c] - vs[f]) * (k - f)

def stats_for(samples, sl):
    run = [s["g"]["running"] for s in samples[sl[0]:sl[1]]]
    wait = [s["g"]["waiting"] for s in samples[sl[0]:sl[1]]]
    kv = [s["g"]["kv_usage"] for s in samples[sl[0]:sl[1]]]
    t0 = samples[sl[0]]["t"]; t1 = samples[sl[1]-1]["t"]
    return run, wait, kv, t0, t1

def summarize(run, wait, kv):
    # modal running
    try:
        mode = st.mode(round(r, 1) for r in run)
        mode_frac = sum(1 for r in run if abs(r - mode) < 0.5) / len(run)
    except Exception:
        mode, mode_frac = float("nan"), float("nan")
    return {
        "run_mean": round(st.mean(run), 2), "run_p50": round(pctl(run, .5), 0),
        "run_p95": round(pctl(run, .95), 0), "run_max": max(run),
        "run_mode": mode, "run_mode_frac": round(mode_frac, 3),
        "wait_mean": round(st.mean(wait), 2), "wait_p50": pctl(wait, .5),
        "wait_p95": pctl(wait, .95), "wait_max": max(wait),
        "wait_gt0_frac": round(sum(1 for w in wait if w > 0) / len(wait), 3),
        "kv_mean": round(st.mean(kv), 4), "kv_p50": round(pctl(kv, .5), 4),
        "kv_p95": round(pctl(kv, .95), 4), "kv_max": round(max(kv), 4),
    }

out = {"note_scheduled_tokens": "gauge schema = {t, g{running,waiting,kv_usage}, wbr{capacity,deferred}} — scheduled_tokens NOT recorded"}
for name in ["ar", "dflash"]:
    raw = load_raw(name)
    with open(f"{BASE}/c_eff_{name}_20260816.json") as f:
        summ = json.load(f)
    pts = summ["points"]
    segs = segment(raw)
    print(f"\n{'='*70}\n=== {name.upper()} : {len(pts)} summary points, {len(segs)} detected segments ===\n{'='*70}")
    out[name] = {"meta": summ["meta"], "n_points": len(pts), "n_segments": len(segs), "points": []}

    # sample interval diagnostics
    dt = [b["t"] - a["t"] for a, b in zip(raw[:-1], raw[1:]) if 0 < b["t"] - a["t"] < 30]
    print(f"sample interval: p50={pctl(dt,.5):.2f}s p95={pctl(dt,.95):.2f}s n={len(raw)}")
    out[name]["sample_interval_p50_s"] = round(pctl(dt, .5), 2)

    hdr = f"{'C':>4} {'seg_s':>7} | {'run_p50':>7} {'run_p95':>7} {'run_max':>7} {'mode':>5} {'mfrac':>5} | {'wait_p50':>8} {'wait_p95':>8} {'wait_max':>8} {'w>0%':>5} | {'kv_p50':>7} {'kv_p95':>7} {'kv_max':>7}"
    print(hdr); print("-" * len(hdr))
    seg_iter = iter(segs)
    seg = next(seg_iter, None)
    for pt in pts:
        # find segment whose duration best matches this point's duration_s
        best = None
        for s0, s1 in segs:
            dur = raw[s1-1]["t"] - raw[s0]["t"]
            if abs(dur - pt["duration_s"]) / pt["duration_s"] < 0.5:
                # first unmatched in order
                best = (s0, s1)
                break
        if best is None:
            print(f"{pt['C']:>4} NO MATCHED SEGMENT (duration_s={pt['duration_s']:.0f})")
            out[name]["points"].append({"C": pt["C"], "matched": False})
            continue
        s0, s1 = best
        run, wait, kv, t0, t1 = stats_for(raw, (s0, s1))
        n = s1 - s0
        ns = int(n * STEADY_FRAC)
        srun, swait, skv = run[ns:], wait[ns:], kv[ns:]
        full, steady = summarize(run, wait, kv), summarize(srun, swait, skv)
        dur = t1 - t0
        print(f"{pt['C']:>4} {dur:>7.0f} | {steady['run_p50']:>7.0f} {steady['run_p95']:>7.0f} {steady['run_max']:>7.0f} {steady['run_mode']:>5.0f} {steady['run_mode_frac']:>5.2f} | {steady['wait_p50']:>8.0f} {steady['wait_p95']:>8.0f} {steady['wait_max']:>8.0f} {steady['wait_gt0_frac']*100:>4.0f}% | {steady['kv_p50']:>7.3f} {steady['kv_p95']:>7.3f} {steady['kv_max']:>7.3f}")
        out[name]["points"].append({
            "C": pt["C"], "matched": True, "seg_duration_s": round(dur, 0),
            "steady": steady, "full": full,
            "burst_only_max": {"run": full["run_max"] > steady["run_max"],
                               "wait": full["wait_max"] > steady["wait_max"],
                               "kv": full["kv_max"] > steady["kv_max"]},
        })

with open(f"{BASE}/e0_steady_state_20260818.json", "w") as f:
    json.dump(out, f, indent=1, ensure_ascii=False)
print("\nsaved:", f"{BASE}/e0_steady_state_20260818.json")
