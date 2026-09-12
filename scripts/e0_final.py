#!/usr/bin/env python3
"""E0 FINAL: steady-state re-analysis of 0816 raw gauge.

Boundary method: cumulative point durations anchored at first active sample.
Validated against running+waiting step detection (16/16 boundaries match +-10s).
Steady state = last 80% of each point (settling excluded).
"""
import json
import gzip
import statistics as st

BASE = "/root/autodl-tmp/litedrafter/outputs"

def pctl(vals, q):
    vs = sorted(vals)
    if not vs:
        return float("nan")
    k = (len(vs) - 1) * q
    f = int(k)
    c = min(f + 1, len(vs) - 1)
    return vs[f] + (vs[c] - vs[f]) * (k - f)

def summarize(run, wait, kv):
    try:
        mode = st.mode(round(r, 1) for r in run)
        mode_frac = sum(1 for r in run if abs(r - mode) < 0.5) / len(run)
    except Exception:
        mode, mode_frac = float("nan"), float("nan")
    return {
        "run_p50": pctl(run, .5), "run_p95": pctl(run, .95), "run_max": max(run),
        "run_mean": st.mean(run), "run_mode": mode, "run_mode_frac": round(mode_frac, 3),
        "wait_p50": pctl(wait, .5), "wait_p95": pctl(wait, .95), "wait_max": max(wait),
        "wait_gt0_frac": round(sum(1 for w in wait if w > 0) / len(wait), 3),
        "kv_p50": pctl(kv, .5), "kv_p95": pctl(kv, .95), "kv_max": max(kv), "kv_mean": st.mean(kv),
    }

out = {"generated": "2026-08-18", "method": "cumulative-duration boundaries, steady=last 80%",
       "scheduled_tokens_recorded": False}
for name in ["ar", "dflash"]:
    with gzip.open(f"{BASE}/c_eff_{name}_raw_20260816.json.gz", "rt") as f:
        raw = json.load(f)
    with open(f"{BASE}/c_eff_{name}_20260816.json") as f:
        summ = json.load(f)
    pts = summ["points"]
    t0 = next(s["t"] for s in raw if s["g"]["running"] > 0 or s["g"]["waiting"] > 0)
    print(f"\n{'='*100}\n=== {name.upper()}  anchor t0, span={raw[-1]['t']-t0:.0f}s vs sum_dur={sum(p['duration_s'] for p in pts):.0f}s ===\n{'='*100}")
    out[name] = {"points": []}
    hdr = (f"{'C':>3} {'dur':>5} {'n':>4} | {'r_p50':>5} {'r_p95':>5} {'r_max':>5} {'r_mod':>5} {'mfr':>4} | "
           f"{'w_p50':>5} {'w_p95':>5} {'w_max':>5} {'w>0%':>5} | {'kv_p50':>6} {'kv_p95':>6} {'kv_max':>6} | {'bmax>rmax':>9}")
    print(hdr); print("-" * len(hdr))
    cursor = t0
    for pt in pts:
        tend = cursor + pt["duration_s"]
        seg = [s for s in raw if cursor <= s["t"] < tend]
        cursor = tend
        if len(seg) < 20:
            print(f"{pt['C']:>3} EMPTY SEGMENT")
            out[name]["points"].append({"C": pt["C"], "empty": True})
            continue
        run = [s["g"]["running"] for s in seg]
        wait = [s["g"]["waiting"] for s in seg]
        kv = [s["g"]["kv_usage"] for s in seg]
        n0 = int(len(seg) * 0.8)
        steady = summarize(run[n0:], wait[n0:], kv[n0:])
        full = summarize(run, wait, kv)
        burst_flag = "YES" if full["run_max"] > steady["run_max"] else "no"
        print(f"{pt['C']:>3} {pt['duration_s']:>5.0f} {len(seg):>4} | "
              f"{steady['run_p50']:>5.0f} {steady['run_p95']:>5.0f} {steady['run_max']:>5.0f} {steady['run_mode']:>5.0f} {steady['run_mode_frac']:>4.2f} | "
              f"{steady['wait_p50']:>5.0f} {steady['wait_p95']:>5.0f} {steady['wait_max']:>5.0f} {steady['wait_gt0_frac']*100:>4.0f}% | "
              f"{steady['kv_p50']:>6.3f} {steady['kv_p95']:>6.3f} {steady['kv_max']:>6.3f} | {burst_flag:>9}")
        out[name]["points"].append({
            "C": pt["C"], "n_samples": len(seg), "steady": {k: round(v, 4) for k, v in steady.items()},
            "full_run_max": full["run_max"], "full_wait_max": full["wait_max"], "full_kv_max": round(full["kv_max"], 4),
            "max_from_burst_only": burst_flag == "YES",
        })

# point-set inventory vs E2 targets
print("\n=== E2 补点缺口盘点 ===")
for name in ["ar", "dflash"]:
    with open(f"{BASE}/c_eff_{name}_20260816.json") as f:
        cs = [p["C"] for p in json.load(f)["points"]]
    print(f"{name}: scanned C = {cs}")
print("E2 targets: dflash C=9/10/11 (ceiling/knee); ar C=104/108/112/116/120/128 (AR knee)")
print("当前 dflash 扫过 C=8,12 → 9/10/11 全缺; ar 扫到 96 → 104-128 全缺")

with open(f"{BASE}/e0_steady_state_20260818.json", "w") as f:
    json.dump(out, f, indent=1, ensure_ascii=False)
print("\nsaved:", f"{BASE}/e0_steady_state_20260818.json")
