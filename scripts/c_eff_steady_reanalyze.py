#!/usr/bin/env python3
"""c_eff_steady_reanalyze.py — 通用稳态重析器 (2026-08-20, 由 AR/n_spec 三轮分析固化)

输入: c_eff_scan.py 落盘的 raw poll 序列 (c_eff_{mode}{tag}_raw_{TODAY}.json.gz,
flat 时序 [{t, g:{running,waiting,kv_usage}, wbr:{capacity,deferred}}, ...])

切段口径 (与 0820 归因报告§7.1一致): 点结束全部请求完成 → running 归零间隙;
每段最后 80% 为稳态。

用法:
  python c_eff_steady_reanalyze.py RAW.json.gz --cs 96,104,108,112,116,120,128
  python c_eff_steady_reanalyze.py RAW.json.gz --probe          # 只打印 drain gaps
  python c_eff_steady_reanalyze.py RAW.json.gz --cs 8,10 --bounds 645,1294   # 显式边界
  python c_eff_steady_reanalyze.py RAW.json.gz --cs 24 --pool 4165           # 反解 blocks/req

坑 (均有对应防护):
  1. 首尾 gap 处理错会把 C 标签整行错位 — 症状: 某段稳态样本个位数 → 打 WARN 并跳过
  2. 高并发点 max_wait 含 chunked-prefill 瞬时排队, 只有稳态(最后80%)才是容量信号
  3. 每段样本数应≈点时长×5 (0.2s 采样), 偏差过大打 WARN
stdlib only, 服务器任意 python 可跑。
"""
import argparse, gzip, json, sys


def load(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        return json.load(f)


def find_gaps(raw, thresh=0):
    """running<=thresh 的连续区间 (drain gaps)"""
    idx = [i for i, p in enumerate(raw) if p["g"]["running"] <= thresh]
    groups = []
    for i in idx:
        if groups and i - groups[-1][-1] <= 3:
            groups[-1].append(i)
        else:
            groups.append([i])
    return groups


def segments_from_gaps(raw, groups, n_points):
    """期望 gap 组数 = 点数: 1 个开头扰动组 + (n-1) 个点间 drain 组; 末段到 EOF。
    (raw 在最后一点结束即停止采样, 文件尾不构成 gap 组)"""
    if len(groups) != n_points:
        return None
    segs = []
    for gi in range(n_points - 1):
        a = groups[gi][-1] + 1
        b = groups[gi + 1][0] - 1
        segs.append((a, b))
    segs.append((groups[-1][-1] + 1, len(raw) - 1))
    return segs


def q(xs, p):
    xs = sorted(xs)
    return xs[min(int(p * len(xs)), len(xs) - 1)]


def analyze(raw, segs, cs, pool):
    print(f"{'C':>4} {'n':>5} {'dur_s':>6} | {'run_p50':>7} {'run_p95':>7} {'run_max':>7} | "
          f"{'wait_p50':>8} {'wait_p95':>8} {'wait_nz%':>8} | {'cap_nz%':>7} | "
          f"{'kv_p50':>6} {'kv_p95':>6} {'kv_max':>6} | {'blk/req':>7}")
    out = {}
    for C, (a, b) in zip(cs, segs):
        seg = raw[a:b + 1]
        if len(seg) < 50:
            print(f"{C:>4}  WARN: 段样本仅 {len(seg)} — 疑似切分错位(标签偏移), 跳过")
            continue
        st = seg[int(len(seg) * 0.2):]
        if len(st) < 10:
            print(f"{C:>4}  WARN: 稳态样本仅 {len(st)} — 疑似切分错位, 跳过")
            continue
        run = [p["g"]["running"] for p in st]
        wait = [p["g"]["waiting"] for p in st]
        kv = [p["g"]["kv_usage"] for p in st]
        cap = [p["wbr"]["capacity"] for p in st]
        nz = lambda xs: round(sum(1 for x in xs if x > 0) / len(xs) * 100, 1)
        dur = st[-1]["t"] - st[0]["t"]
        if dur > 0 and not (0.7 * dur * 5 <= len(st) <= 1.4 * dur * 5):
            print(f"       (warn: 样本数 {len(st)} vs 时长×5={dur*5:.0f} 偏离)")
        bpr = round(q(kv, .5) * pool / q(run, .5), 1) if pool and q(run, .5) else ""
        print(f"{C:>4} {len(st):>5} {dur:>6.0f} | {q(run,.5):>7.0f} {q(run,.95):>7.0f} {max(run):>7.0f} | "
              f"{q(wait,.5):>8.0f} {q(wait,.95):>8.0f} {nz(wait):>7.1f}% | {nz(cap):>6.1f}% | "
              f"{q(kv,.5):>6.3f} {q(kv,.95):>6.3f} {max(kv):>6.3f} | {bpr:>7}")
        out[C] = {"running_p50_p95_max": [q(run, .5), q(run, .95), max(run)],
                  "waiting_p50_p95": [q(wait, .5), q(wait, .95)],
                  "waiting_nonzero_pct": nz(wait), "capacity_nonzero_pct": nz(cap),
                  "kv_p50_p95_max": [q(kv, .5), q(kv, .95), max(kv)],
                  "n_steady_samples": len(st)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("--cs", help="逗号分隔的每点 C 值(按时间序)")
    ap.add_argument("--pool", type=float, default=0, help="usable pool blocks, 反解 blocks/req")
    ap.add_argument("--probe", action="store_true", help="只打印 drain gaps")
    ap.add_argument("--bounds", help="显式段尾边界 idx(逗号分隔, 个数=点数), 防自动切分错位")
    ap.add_argument("-o", "--output", help="结果 json 落盘路径")
    args = ap.parse_args()

    raw = load(args.raw)
    t0 = raw[0]["t"]
    groups = find_gaps(raw)
    print(f"raw points: {len(raw)}, dur {raw[-1]['t']-t0:.0f}s, drain gaps: {len(groups)}")
    for g in groups:
        print(f"  gap idx[{g[0]}:{g[-1]}] t={raw[g[0]]['t']-t0:7.1f}s")

    if args.probe:
        return
    if not args.cs:
        sys.exit("need --cs (comma list) or --probe")
    cs = [int(x) for x in args.cs.split(",") if x.strip()]

    if args.bounds:
        bnds = [int(x) for x in args.bounds.split(",")]
        assert len(bnds) == len(cs), "bounds 数应=点数"
        segs, prev = [], 0
        for b in bnds:
            segs.append((prev, b)); prev = b + 1
    else:
        segs = segments_from_gaps(raw, groups, len(cs))
        if segs is None:
            sys.exit(f"gap 数 {len(groups)} != 点数+1 {len(cs)+1} — 用 --probe 查看后以 --bounds 显式给定")

    out = analyze(raw, segs, cs, args.pool)
    if args.output:
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"saved -> {args.output}")


if __name__ == "__main__":
    main()
