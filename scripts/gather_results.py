#!/usr/bin/env python3
"""
gather_results.py - 汇总 Phase 1.2 sweep 结果

输入: outputs/p1b_*.json
输出:
  outputs/p1_2_summary_table.csv
  outputs/p1_2_oom_boundary.md
  outputs/p1_2_acceptance_histogram.csv
  figs/p1_2_max_batch.png
  figs/p1_2_peak_reserved.png
  figs/p1_2_throughput.png
  figs/p1_2_acceptance_dist.png
"""

import json
import os
import sys
from glob import glob

OUTPUT_DIR = "/root/autodl-tmp/litedrafter/outputs"
FIGS_DIR = os.path.join(OUTPUT_DIR, "..", "figs")
VARIANTS = ["raw", "ctx1024", "ctx2048", "ctx4096"]
MODES = ["ar", "dflash"]


def load_results():
    """Load all p1b_*.json files, return nested dict."""
    results = {}
    pattern = os.path.join(OUTPUT_DIR, "p1b_*.json")
    for fpath in sorted(glob(pattern)):
        with open(fpath) as f:
            r = json.load(f)
        key = (r["mode"], r["variant"], r["batch_size"], r["run_id"])
        results[key] = r
    return results


def avg_of_runs(results, mode, variant, batch_size):
    """Average metrics across repeated runs."""
    runs = []
    for rid in [1, 2, 3]:
        k = (mode, variant, batch_size, rid)
        if k in results and results[k].get("success"):
            runs.append(results[k])
    if not runs:
        return None

    return {
        "num_runs": len(runs),
        "peak_allocated_mb": sum(r["peak_allocated_mb"] for r in runs) / len(runs),
        "peak_reserved_mb": sum(r["peak_reserved_mb"] for r in runs) / len(runs),
        "total_latency_s": sum(r["total_latency_s"] for r in runs) / len(runs),
        "total_wall_s": sum(r["total_wall_s"] for r in runs) / len(runs),
        "num_output_tokens": sum(r["num_output_tokens"] for r in runs) / len(runs),
        "throughput_tok_s": sum(r["throughput_tok_s"] for r in runs) / len(runs),
        "tpot_ms": sum(r["tpot_ms"] for r in runs) / len(runs),
        "headroom_mb": sum(r["headroom_mb"] for r in runs) / len(runs),
    }


def write_csv(results):
    """Summary table CSV."""
    path = os.path.join(OUTPUT_DIR, "p1_2_summary_table.csv")
    with open(path, "w") as f:
        hdr = ("mode,variant,batch_size,num_runs,success,oom,"
               "peak_allocated_mb,peak_reserved_mb,total_wall_s,"
               "num_output_tokens,throughput_tok_s,tpot_ms,headroom_mb,"
               "speedup,used_acc_mean,raw_acc_mean")
        f.write(hdr + "\n")

        rows = []
        for variant in VARIANTS:
            for mode in MODES:
                for bs in [1, 2, 4, 8, 16, 32]:
                    k = (mode, variant, bs, 1)
                    if k not in results:
                        continue
                    r = results[k]
                    avg = avg_of_runs(results, mode, variant, bs)
                    if avg is None:
                        continue

                    # speedup vs AR same variant same batch
                    ar_avg = avg_of_runs(results, "ar", variant, bs)
                    speedup = ""
                    if ar_avg and mode == "dflash":
                        speedup = f"{ar_avg['throughput_tok_s'] / max(avg['throughput_tok_s'], 0.01):.1f}x"

                    acc = r.get("acceptance_stats", {})
                    line = (f"{mode},{variant},{bs},{avg['num_runs']},"
                            f"{r.get('success',False)},{r.get('oom',False)},"
                            f"{avg['peak_allocated_mb']:.0f},{avg['peak_reserved_mb']:.0f},"
                            f"{avg['total_wall_s']:.1f},{avg['num_output_tokens']:.0f},"
                            f"{avg['throughput_tok_s']:.1f},{avg['tpot_ms']:.1f},"
                            f"{avg['headroom_mb']:.0f},{speedup},"
                            f"{acc.get('used_mean','')},{acc.get('raw_mean','')}")
                    f.write(line + "\n")
    print(f"[SAVED] {path}")


def write_oom_boundary(results):
    """OOM boundary markdown."""
    path = os.path.join(OUTPUT_DIR, "p1_2_oom_boundary.md")
    with open(path, "w") as f:
        f.write("# Phase 1.2 OOM Boundary\n\n")
        f.write("| Mode | Variant | Max Batch | OOM At | Peak at Max (reserved MB) | Headroom at Max (MB) |\n")
        f.write("|------|---------|-----------|--------|---------------------------|----------------------|\n")
        for mode in MODES:
            for variant in VARIANTS:
                max_b = None
                oom_at = None
                for bs in [1, 2, 4, 8, 16, 32]:
                    k = (mode, variant, bs, 1)
                    if k in results:
                        r = results[k]
                        if r.get("success"):
                            max_b = bs
                        elif r.get("oom"):
                            oom_at = bs
                            break
                if max_b:
                    avg = avg_of_runs(results, mode, variant, max_b)
                    pk = avg["peak_reserved_mb"] if avg else 0
                    hr = avg["headroom_mb"] if avg else 0
                    f.write(f"| {mode} | {variant} | {max_b} | {oom_at or '-'} | {pk:.0f} | {hr:.0f} |\n")
    print(f"[SAVED] {path}")


def write_acceptance_histogram(results):
    """Aggregate DFlash acceptance data across all runs."""
    path = os.path.join(OUTPUT_DIR, "p1_2_acceptance_histogram.csv")
    with open(path, "w") as f:
        f.write("variant,batch_size,run_id,step_idx,used,raw\n")
        for variant in VARIANTS:
            for bs in [1, 2, 4, 8, 16, 32]:
                for rid in [1, 2, 3]:
                    k = ("dflash", variant, bs, rid)
                    if k not in results or not results[k].get("success"):
                        continue
                    r = results[k]
                    acc_raw = r.get("acceptance_raw", [])
                    for si, acc in enumerate(acc_raw):
                        used = acc.get("used", 0)
                        raw_vals = ",".join(str(v) for v in acc.get("raw", []))
                        f.write(f"{variant},{bs},{rid},{si},{used},{raw_vals}\n")
    print(f"[SAVED] {path}")


def make_figures(results):
    """Generate 4 matplotlib figures."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not installed, skipping figures")
        return

    os.makedirs(FIGS_DIR, exist_ok=True)

    # Figure 1: max successful batch
    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(VARIANTS))
    width = 0.35
    for i, mode in enumerate(MODES):
        max_batches = []
        for variant in VARIANTS:
            mb = None
            for bs in [32, 16, 8, 4, 2, 1]:
                k = (mode, variant, bs, 1)
                if k in results and results[k].get("success"):
                    mb = bs
                    break
            max_batches.append(mb or 0)
        ax.bar([xi + i*width for xi in x], max_batches, width, label=mode)
    ax.set_xticks([xi + width/2 for xi in x])
    ax.set_xticklabels(VARIANTS)
    ax.set_ylabel("Max Successful Batch")
    ax.set_title("Max Batch per Mode × Variant")
    ax.legend()
    fig.savefig(os.path.join(FIGS_DIR, "p1_2_max_batch.png"), dpi=150)
    plt.close(fig)
    print(f"[SAVED] figs/p1_2_max_batch.png")

    # Figure 2: peak_reserved vs batch
    fig, ax = plt.subplots(figsize=(12, 6))
    for variant in VARIANTS:
        for mode in MODES:
            xs, ys = [], []
            for bs in [1, 2, 4, 8, 16, 32]:
                avg = avg_of_runs(results, mode, variant, bs)
                if avg:
                    xs.append(bs)
                    ys.append(avg["peak_reserved_mb"])
            if xs:
                ax.plot(xs, ys, marker="o", label=f"{mode}_{variant}")
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Peak Reserved (MB)")
    ax.set_title("Peak Reserved vs Batch")
    ax.legend(fontsize=8)
    ax.grid(True)
    fig.savefig(os.path.join(FIGS_DIR, "p1_2_peak_reserved.png"), dpi=150)
    plt.close(fig)
    print(f"[SAVED] figs/p1_2_peak_reserved.png")

    # Figure 3: throughput vs batch
    fig, ax = plt.subplots(figsize=(12, 6))
    for variant in VARIANTS:
        for mode in MODES:
            xs, ys = [], []
            for bs in [1, 2, 4, 8, 16, 32]:
                avg = avg_of_runs(results, mode, variant, bs)
                if avg:
                    xs.append(bs)
                    ys.append(avg["throughput_tok_s"])
            if xs:
                ax.plot(xs, ys, marker="s", label=f"{mode}_{variant}")
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Throughput (tokens/s)")
    ax.set_title("Throughput vs Batch")
    ax.legend(fontsize=8)
    ax.grid(True)
    fig.savefig(os.path.join(FIGS_DIR, "p1_2_throughput.png"), dpi=150)
    plt.close(fig)
    print(f"[SAVED] figs/p1_2_throughput.png")

    # Figure 4: accepted length distribution
    all_used = []
    for variant in VARIANTS:
        for bs in [1, 2, 4, 8, 16, 32]:
            for rid in [1, 2, 3]:
                k = ("dflash", variant, bs, rid)
                if k not in results or not results[k].get("success"):
                    continue
                r = results[k]
                for acc in r.get("acceptance_raw", []):
                    all_used.append(acc.get("used", 0))

    if all_used:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(all_used, bins=20, edgecolor="black")
        ax.set_xlabel("Used Acceptance Length")
        ax.set_ylabel("Frequency")
        ax.set_title("DFlash Acceptance Length Distribution (all runs)")
        fig.savefig(os.path.join(FIGS_DIR, "p1_2_acceptance_dist.png"), dpi=150)
        plt.close(fig)
        print(f"[SAVED] figs/p1_2_acceptance_dist.png")


if __name__ == "__main__":
    results = load_results()
    print(f"Loaded {len(results)} result files")

    write_csv(results)
    write_oom_boundary(results)
    write_acceptance_histogram(results)
    make_figures(results)
    print("\n[DONE] All outputs generated")
