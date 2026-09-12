#!/usr/bin/env python3
"""e3_analyze.py — E3 ledger 分析:Q1 绑定组 / Q2 每组实持 / 502 闭合"""
import json, statistics as st
from collections import Counter

LED = "/root/autodl-tmp/litedrafter/outputs/e3_ledger_dflash_20260818_1123.jsonl"
recs = [json.loads(l) for l in open(LED)]
gt = [r for r in recs if r["ev"] == "group_table"][0]
gtypes = {g["gid"]: (g["type"], g["block_size"]) for g in gt["groups"]}

# ---- Q1: admit_fail 汇总 ----
fails = [r for r in recs if r["ev"] == "admit_fail"]
f = fails[-1]
mamba = sum(d["need"] for d in f["decomp"] if "Mamba" in d["type"])
attn = sum(d["need"] for d in f["decomp"]) - mamba
print("== Q1 admit_fail 样例(fail_n=1) ==")
print(f"  ntok={f['ntok']} new={f['new']} lookahead={f['lookahead']} pool_free={f['pool_free']}")
print(f"  总need={sum(d['need'] for d in f['decomp'])} = mamba {mamba} ({mamba/(mamba+attn):.0%}) + attn {attn} ({attn/(mamba+attn):.0%})")
print(f"  mamba明细: 24组×{mamba//24}块(常数)  attn明细: 14组×2块(cdiv({f['ntok']}+16,592)=2)")

# ---- Q2: finish 每组持块 ----
fins = [r for r in recs if r["ev"] == "finish"]
full = [r for r in fins if r.get("ntok", 0) >= 1260]
print(f"\n== Q2 finish 持块 ==\n完整请求 {len(full)}/{len(fins)}  ntok分布: {Counter(r['ntok'] for r in fins).most_common(3)}")

n = 38
agg = {gid: [] for gid in range(n)}
for r in full:
    if len(r["held"]) == n:
        for gid in range(n):
            agg[gid].append(r["held"][gid])

byt = {}
for gid in range(n):
    if agg[gid]:
        t, bs = gtypes[gid]
        byt.setdefault((t, bs), []).append((gid, st.median(agg[gid]), max(agg[gid])))
for (t, bs), rows in sorted(byt.items()):
    med = sum(x[1] for x in rows) / len(rows)
    detail = " ".join("g%d:%.0f" % (g, m) for g, m, _ in rows[:6])
    print(f"  {t} bs={bs}: {len(rows)}组 p50均值={med:.1f} | {detail} ...")

tots = [sum(r["held"]) for r in full if len(r["held"]) == n]
if tots:
    print(f"\n每请求总持块: p50={st.median(tots):.0f} mean={st.mean(tots):.1f} max={max(tots)}")
    print(f"9请求稳态合计≈{9*st.median(tots):.0f} + 剩余free vs 池3919 → 与admit_fail时pool_free={f['pool_free']}对账")

# ---- admit_ok 验证 ----
oks = [r for r in recs if r["ev"] == "admit_ok"]
o = oks[len(oks)//2]
mamba_o = sum(d["need"] for d in o["decomp"] if "Mamba" in d["type"])
print(f"\n== admit_ok 样例 == ntok={o['ntok']} new={o['new']} look={o['lookahead']} pool_free={o['pool_free']}")
print(f"  总need={sum(d['need'] for d in o['decomp'])} (mamba={mamba_o}, attn={sum(d['need'] for d in o['decomp'])-mamba_o})")

# ---- 502 分解闭合 ----
print("\n== 502 分解 ==")
mamba_hold = 384
attn_hold = sum(st.median(agg[gid]) for gid in range(24, 38))
print(f"  mamba稳态: 24组×16 = {mamba_hold}")
print(f"  attn稳态: 14组×~{attn_hold/14:.0f} = {attn_hold:.0f} (cdiv(1268+16,592)=3)")
print(f"  每请求稳态 ≈ {mamba_hold+attn_hold:.0f} blocks (E1离线上界1294的差距=attn按592tok/block非16)")
print(f"  runtime反解 4039/9={4039/9:.0f} ↔ 实测p50 {st.median(tots):.0f}")
